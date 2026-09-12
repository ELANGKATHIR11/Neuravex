
"""
YOLO27 v0.4 multi-task training system.

Designed to combine:
- COCO detection/classification
- COCO instance segmentation
- dense DEM/depth when available
- metric 3D geometry when available
- boundary supervision
- augmentation consistency
- learned uncertainty-based adaptive task weighting

COCO's segmentation JSON supplies 2D instance masks; it does not supply metric depth,
L/W/H, camera-coordinate 3D boxes, or poses. Missing modalities are therefore masked
out of the corresponding losses rather than fabricated.
"""
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- Geometry -------------------------------------------------

def box_iou_2d_xyxy(a,b,eps=1e-7):
    lt=torch.maximum(a[...,:2],b[...,:2])
    rb=torch.minimum(a[...,2:],b[...,2:])
    wh=(rb-lt).clamp(min=0)
    inter=wh[...,0]*wh[...,1]
    area_a=((a[...,2]-a[...,0]).clamp(min=0)*
            (a[...,3]-a[...,1]).clamp(min=0))
    area_b=((b[...,2]-b[...,0]).clamp(min=0)*
            (b[...,3]-b[...,1]).clamp(min=0))
    return inter/(area_a+area_b-inter+eps)

def boxes3d_corners(center,lwh,yaw):
    """Returns 8 corners for yaw-oriented boxes in camera/world coordinates."""
    B=center.shape[0]
    l,w,h=lwh[:,0],lwh[:,1],lwh[:,2]
    x=torch.stack([ l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2],1)
    y=torch.stack([ w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2],1)
    z=torch.stack([-h/2,-h/2,-h/2,-h/2, h/2, h/2, h/2, h/2],1)
    c,s=torch.cos(yaw).view(-1,1),torch.sin(yaw).view(-1,1)
    xr=c*x-s*y; yr=s*x+c*y
    return torch.stack([xr+center[:,0:1],yr+center[:,1:2],z+center[:,2:3]],-1)

def aabb_from_corners(c):
    return c.min(1).values,c.max(1).values

def iou3d_aabb(pred_corners, gt_corners, eps=1e-7):
    """
    Differentiable conservative 3D overlap using enclosing AABBs.
    For a yaw-oriented exact IoU kernel, replace this with polygon intersection
    + vertical overlap. This AABB objective remains useful as an auxiliary loss.
    """
    p0,p1=aabb_from_corners(pred_corners); g0,g1=aabb_from_corners(gt_corners)
    lo=torch.maximum(p0,g0); hi=torch.minimum(p1,g1)
    inter=(hi-lo).clamp(min=0).prod(-1)
    pv=(p1-p0).clamp(min=0).prod(-1); gv=(g1-g0).clamp(min=0).prod(-1)
    return inter/(pv+gv-inter+eps)

def giou3d_aabb(pred_corners,gt_corners,eps=1e-7):
    p0,p1=aabb_from_corners(pred_corners); g0,g1=aabb_from_corners(gt_corners)
    lo=torch.minimum(p0,g0); hi=torch.maximum(p1,g1)
    enc=(hi-lo).clamp(min=0).prod(-1)
    iou=iou3d_aabb(pred_corners,gt_corners,eps)
    pvol=(p1-p0).clamp(min=0).prod(-1)
    gvol=(g1-g0).clamp(min=0).prod(-1)
    # union recovered from IoU relation
    inter=iou*(pvol+gvol)/(1+iou+eps)
    union=pvol+gvol-inter
    return iou-(enc-union)/(enc+eps)

# ---------- Segmentation losses ------------------------------------

def dice_loss(logits,target,eps=1e-6):
    p=torch.sigmoid(logits)
    t=target.float()
    dims=tuple(range(2,p.ndim))
    inter=(p*t).sum(dims)
    den=p.sum(dims)+t.sum(dims)
    return (1-(2*inter+eps)/(den+eps)).mean()

def binary_focal_loss(logits,target,alpha=.25,gamma=2.0):
    bce=F.binary_cross_entropy_with_logits(logits,target.float(),reduction="none")
    p=torch.sigmoid(logits)
    pt=p*target+(1-p)*(1-target)
    a=alpha*target+(1-alpha)*(1-target)
    return (a*(1-pt).pow(gamma)*bce).mean()

def semantic_loss(logits,target):
    return F.cross_entropy(logits,target.long()) + dice_loss(logits,target.unsqueeze(1).float().expand_as(logits))

def boundary_from_mask(mask):
    """Approximate boundary target using morphological gradient."""
    m=mask.float().unsqueeze(1)
    dil=F.max_pool2d(m,3,1,1)
    ero=-F.max_pool2d(-m,3,1,1)
    return (dil-ero).clamp(0,1)

def boundary_loss(pred,target):
    return binary_focal_loss(pred,target)+dice_loss(pred,target)

def instance_embedding_loss(emb,instance_ids,margin=.5):
    """
    Pull pixels belonging to one instance toward its centroid and push
    different instance centroids apart. Memory-safe approximation.
    """
    total=[]; B=emb.shape[0]
    for b in range(B):
        ids=instance_ids[b].unique()
        ids=ids[ids>0]
        if len(ids)<1: continue
        centers=[]
        pull=emb.new_tensor(0.)
        for iid in ids:
            m=instance_ids[b]==iid
            if m.sum()<2: continue
            e=emb[b,:,m].T
            c=e.mean(0,keepdim=True)
            centers.append(c.squeeze(0))
            pull=pull+((e-c)**2).mean()
        pull=pull/max(len(ids),1)
        push=emb.new_tensor(0.)
        if len(centers)>1:
            C=torch.stack(centers)
            d=torch.cdist(C,C)
            eye=torch.eye(len(centers),device=emb.device,dtype=torch.bool)
            push=F.relu(margin-d[~eye]).pow(2).mean()
        total.append(pull+push)
    return torch.stack(total).mean() if total else emb.sum()*0

# ---------- Depth/3D losses ----------------------------------------

def scale_invariant_log_depth_loss(pred,gt,valid):
    p=torch.log(pred.clamp_min(1e-4)); g=torch.log(gt.clamp_min(1e-4))
    v=valid.bool()
    if v.sum()==0: return pred.sum()*0
    d=(p-g)[v]
    return (d*d).mean()-0.5*d.mean().pow(2)

def uncertainty_l1(pred,target,log_sigma,valid=None):
    sigma=torch.exp(log_sigma).view(1,-1,*([1]*(pred.ndim-2))).clamp_min(1e-4)
    e=(pred-target).abs()
    if valid is not None: e=e*valid
    loss=e/sigma+torch.log(sigma)
    if valid is not None:
        return loss.sum()/(valid.sum()*pred.shape[1]+1e-6)
    return loss.mean()

def lwh_loss(pred_log,target,log_sigma,valid=None):
    return uncertainty_l1(torch.exp(pred_log),target,log_sigma,valid)

def yaw_loss(pred_sc,target,log_sigma,valid=None):
    p=F.normalize(pred_sc,dim=1)
    t=torch.cat([torch.sin(target),torch.cos(target)],dim=1)
    e=1-(p*t).sum(1,keepdim=True)
    sigma=torch.exp(log_sigma).view(1,1,*([1]*(e.ndim-2))).clamp_min(1e-4)
    if valid is not None: e=e*valid
    return (e/sigma+torch.log(sigma)).mean()

def geometry_loss(out,t):
    """Combines center, dimension, yaw, corners and conservative 3D IoU."""
    valid=t.get("valid_3d")
    if valid is None: return out["xyz"].sum()*0
    lxyz=uncertainty_l1(out["xyz"],t["xyz"],out["log_sigma_xyz"],valid)
    llwh=lwh_loss(out["lwh_log"],t["lwh"],out["log_sigma_lwh"],valid)
    lyaw=yaw_loss(out["yaw_sincos"],t["yaw"],out["log_sigma_yaw"],valid)
    pred_c=boxes3d_corners(out["xyz"].flatten(2).transpose(1,2).reshape(-1,3),
                           torch.exp(out["lwh_log"]).flatten(2).transpose(1,2).reshape(-1,3),
                           torch.atan2(out["yaw_sincos"][:,0],out["yaw_sincos"][:,1]).flatten())
    gt_c=boxes3d_corners(t["xyz"].flatten(2).transpose(1,2).reshape(-1,3),
                         t["lwh"].flatten(2).transpose(1,2).reshape(-1,3),
                         t["yaw"].flatten())
    v=valid.flatten().bool()
    if v.any():
        liou=(1-iou3d_aabb(pred_c[v],gt_c[v])).mean()
        lgiou=(1-giou3d_aabb(pred_c[v],gt_c[v])).mean()
    else: liou=lgiou=out["xyz"].sum()*0
    return lxyz+llwh+lyaw+0.5*liou+0.25*lgiou

# ---------- Consistency --------------------------------------------

def consistency_loss(a,b,keys=("semantic_masks","boundary_map","depth_inverse")):
    total=a["class_logits"].sum()*0
    for k in keys:
        x,y=a[k],b[k]
        if x.shape!=y.shape: y=F.interpolate(y,size=x.shape[-2:],mode="bilinear",align_corners=False)
        total=total+F.smooth_l1_loss(x,y.detach())
    # normalized embeddings are especially suitable for consistency
    total=total+F.smooth_l1_loss(a["instance_embeddings"],b["instance_embeddings"].detach())
    return total

# ---------- Adaptive task weighting --------------------------------

class AdaptiveTaskWeights(nn.Module):
    """
    Learnable uncertainty weighting:
      L_total = sum_i exp(-s_i) L_i + s_i
    """
    def __init__(self,tasks):
        super().__init__()
        self.tasks=list(tasks)
        self.log_vars=nn.ParameterDict({k:nn.Parameter(torch.zeros(())) for k in self.tasks})
    def forward(self,losses):
        total=0
        weighted={}
        for k in self.tasks:
            if k not in losses: continue
            s=self.log_vars[k]
            w=torch.exp(-s)
            weighted[k]=w*losses[k]+s
            total=total+weighted[k]
        return total,weighted

# ---------- COCO dataset adapter -----------------------------------

class COCOSegDataset(Dataset):
    """
    Lightweight adapter specification. Requires pycocotools.
    It returns image + class masks + instance IDs + boxes.
    The trainer can optionally receive depth/3D targets from another dataset adapter.
    """
    def __init__(self, image_dir, annotation_json, transforms=None):
        from pycocotools.coco import COCO
        self.coco=COCO(annotation_json)
        self.image_dir=image_dir
        self.ids=self.coco.getImgIds()
        self.transforms=transforms
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        info=self.coco.loadImgs([self.ids[i]])[0]
        anns=self.coco.loadAnns(self.coco.getAnnIds(imgIds=[self.ids[i]],iscrowd=None))
        # Full image loading/letterbox/mask rasterization belongs here in production.
        # Kept explicit rather than hiding geometry transforms.
        return {"image_path":str(self.image_dir+"/"+info["file_name"]),
                "image_id":self.ids[i],"annotations":anns,
                "width":info["width"],"height":info["height"]}

# ---------- Trainer skeleton ---------------------------------------

class YOLO27MultiTaskTrainer:
    def __init__(self,model,optimizer,tasks=None,device="cuda"):
        self.model=model.to(device); self.optimizer=optimizer; self.device=device
        tasks=tasks or ["det","semantic","instance","boundary","part","depth","geometry","consistency"]
        self.weights=AdaptiveTaskWeights(tasks).to(device)
        # Include task-weight parameters in optimizer.
        self.optimizer.add_param_group({"params":self.weights.parameters(),"lr":optimizer.param_groups[0]["lr"]})

    def compute_losses(self,out,tgt,aug_out=None):
        losses={}
        if "det_targets" in tgt:
            # Placeholder: attach an OTA/SimOTA-style assignment implementation here.
            losses["det"]=tgt["det_loss"]
        if "semantic_target" in tgt:
            losses["semantic"]=semantic_loss(out["semantic_masks"],tgt["semantic_target"])
        if "instance_ids" in tgt:
            losses["instance"]=instance_embedding_loss(out["instance_embeddings"],tgt["instance_ids"])
        if "boundary_target" in tgt:
            losses["boundary"]=boundary_loss(out["boundary_map"],tgt["boundary_target"])
        if "part_target" in tgt:
            losses["part"]=semantic_loss(out["part_masks"],tgt["part_target"])
        if "depth" in tgt:
            losses["depth"]=scale_invariant_log_depth_loss(
                out["depth_inverse"],tgt["depth"],tgt["depth_valid"])
        if "valid_3d" in tgt:
            losses["geometry"]=geometry_loss(out,tgt)
        if aug_out is not None:
            losses["consistency"]=consistency_loss(out,aug_out)
        return losses

    def step(self,losses):
        total,weighted=self.weights(losses)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.weights.parameters()),10.0)
        self.optimizer.step()
        return total.detach(),{k:v.detach() for k,v in weighted.items()}
