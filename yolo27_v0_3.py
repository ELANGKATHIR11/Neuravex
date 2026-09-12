
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self,c1,c2,k=3,s=1,g=1):
        super().__init__()
        self.m=nn.Sequential(
            nn.Conv2d(c1,c2,k,s,k//2,groups=g,bias=False),
            nn.BatchNorm2d(c2),nn.SiLU(inplace=True))
    def forward(self,x): return self.m(x)

class SE(nn.Module):
    def __init__(self,c,r=8):
        super().__init__()
        h=max(c//r,8)
        self.m=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,h,1),
                             nn.SiLU(inplace=True),nn.Conv2d(h,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.m(x)

class HybridBlock(nn.Module):
    def __init__(self,c,e=2):
        super().__init__()
        h=c*e
        self.a=ConvBNAct(c,h,1)
        self.b=ConvBNAct(h,h,3,1,h)
        self.se=SE(h)
        self.c=nn.Sequential(nn.Conv2d(h,c,1,bias=False),nn.BatchNorm2d(c))
    def forward(self,x): return F.silu(x+self.c(self.se(self.b(self.a(x)))))

class Backbone(nn.Module):
    def __init__(self,b=48):
        super().__init__()
        self.stem=nn.Sequential(ConvBNAct(3,b,3,2),ConvBNAct(b,b,3))
        self.s2=nn.Sequential(ConvBNAct(b,b*2,3,2),HybridBlock(b*2),HybridBlock(b*2))
        self.s3=nn.Sequential(ConvBNAct(b*2,b*4,3,2),HybridBlock(b*4),HybridBlock(b*4),HybridBlock(b*4))
        self.s4=nn.Sequential(ConvBNAct(b*4,b*8,3,2),HybridBlock(b*8),HybridBlock(b*8),HybridBlock(b*8))
        self.s5=nn.Sequential(ConvBNAct(b*8,b*16,3,2),HybridBlock(b*16),HybridBlock(b*16))
    def forward(self,x):
        x=self.stem(x); p2=self.s2(x); p3=self.s3(p2); p4=self.s4(p3); p5=self.s5(p4)
        return p3,p4,p5

class FPN(nn.Module):
    def __init__(self,b=48):
        super().__init__()
        c3,c4,c5=b*4,b*8,b*16
        self.a=ConvBNAct(c5,c4,1)
        self.b=ConvBNAct(c4+c4,c4,3)
        self.c=ConvBNAct(c4,c3,1)
        self.d=ConvBNAct(c3+c3,c3,3)
    def forward(self,p3,p4,p5):
        q5=self.a(p5)
        q4=self.b(torch.cat([F.interpolate(q5,size=p4.shape[-2:],mode="nearest"),p4],1))
        q3=self.d(torch.cat([F.interpolate(self.c(q4),size=p3.shape[-2:],mode="nearest"),p3],1))
        return q3,q4,q5

class DEM(nn.Module):
    def __init__(self,b=48):
        super().__init__()
        c3,c4=b*4,b*8
        self.a=ConvBNAct(c3,b*2,3)
        self.b=ConvBNAct(c4,b*2,3)
        self.c=ConvBNAct(c4,b*2,1)
        self.out=nn.Sequential(ConvBNAct(b*6,b*2,3),nn.Conv2d(b*2,1,1))
    def forward(self,q3,q4,q5,hw):
        a=self.a(q3)
        b=F.interpolate(self.b(q4),size=q3.shape[-2:],mode="bilinear",align_corners=False)
        c=F.interpolate(self.c(q5),size=q3.shape[-2:],mode="bilinear",align_corners=False)
        return F.softplus(F.interpolate(self.out(torch.cat([a,b,c],1)),size=hw,mode="bilinear",align_corners=False))+1e-4

class MultiLayerSegmentation(nn.Module):
    """
    Four native segmentation layers:
      semantic: class logits
      instance: per-pixel embedding vectors
      boundary: object-boundary probability
      part: fine-grained part/detail logits
    """
    def __init__(self,b=48,nc=80,embed=16,parts=16):
        super().__init__()
        c=b*4
        self.f3=ConvBNAct(c,b*2,3)
        self.f4=ConvBNAct(b*8,b*2,3)
        self.f5=ConvBNAct(b*8,b*2,3)
        self.fuse=nn.Sequential(ConvBNAct(b*6,b*2,3),HybridBlock(b*2))
        self.semantic=nn.Conv2d(b*2,nc,1)
        self.instance=nn.Conv2d(b*2,embed,1)
        self.boundary=nn.Conv2d(b*2,1,1)
        self.parts=nn.Conv2d(b*2,parts,1)
        self.mask_quality=nn.Conv2d(b*2,1,1)
    def forward(self,q3,q4,q5,out_hw):
        a=self.f3(q3)
        b=F.interpolate(self.f4(q4),size=q3.shape[-2:],mode="bilinear",align_corners=False)
        c=F.interpolate(self.f5(q5),size=q3.shape[-2:],mode="bilinear",align_corners=False)
        z=self.fuse(torch.cat([a,b,c],1))
        return {
          "semantic_masks":F.interpolate(self.semantic(z),size=out_hw,mode="bilinear",align_corners=False),
          "instance_embeddings":F.normalize(F.interpolate(self.instance(z),size=out_hw,mode="bilinear",align_corners=False),dim=1),
          "boundary_map":torch.sigmoid(F.interpolate(self.boundary(z),size=out_hw,mode="bilinear",align_corners=False)),
          "part_masks":F.interpolate(self.parts(z),size=out_hw,mode="bilinear",align_corners=False),
          "mask_quality":torch.sigmoid(F.interpolate(self.mask_quality(z),size=out_hw,mode="bilinear",align_corners=False))
        }

class CrossTaskFusion(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.g=nn.Sequential(nn.Conv2d(c*3,c,1,bias=False),nn.BatchNorm2d(c),nn.Sigmoid())
    def forward(self,det,seg,depth):
        gate=self.g(torch.cat([det,seg,depth],1))
        return det*(1+gate)

class Heads(nn.Module):
    def __init__(self,b=48,nc=80):
        super().__init__()
        c=b*4
        self.det=nn.Sequential(ConvBNAct(c,c,3),HybridBlock(c))
        self.segfeat=ConvBNAct(c,c,3)
        self.depthfeat=ConvBNAct(c,c,3)
        self.fuse=CrossTaskFusion(c)
        self.cls=nn.Conv2d(c,nc,1); self.obj=nn.Conv2d(c,1,1); self.box=nn.Conv2d(c,4,1)
        self.xyz=nn.Conv2d(c,3,1); self.lwh=nn.Conv2d(c,3,1); self.yaw=nn.Conv2d(c,2,1)
        self.log_sigma_xyz=nn.Parameter(torch.zeros(3))
        self.log_sigma_lwh=nn.Parameter(torch.zeros(3))
        self.log_sigma_yaw=nn.Parameter(torch.zeros(1))
    def forward(self,f):
        d=self.det(f); s=self.segfeat(f); dep=self.depthfeat(f)
        g=self.fuse(d,s,dep)
        return {"class_logits":self.cls(g),"objectness":self.obj(g),"box_raw":self.box(g),
                "xyz":self.xyz(g),"lwh_log":self.lwh(g),"yaw_sincos":self.yaw(g),
                "log_sigma_xyz":self.log_sigma_xyz,"log_sigma_lwh":self.log_sigma_lwh,
                "log_sigma_yaw":self.log_sigma_yaw}

class YOLO27(nn.Module):
    def __init__(self,nc=80,base=48,seg_embed=16,parts=16):
        super().__init__()
        self.backbone=Backbone(base); self.neck=FPN(base)
        self.dem=DEM(base)
        self.seg=MultiLayerSegmentation(base,nc,seg_embed,parts)
        self.head=Heads(base,nc)
    def forward(self,x):
        p3,p4,p5=self.backbone(x); q3,q4,q5=self.neck(p3,p4,p5)
        depth=self.dem(q3,q4,q5,x.shape[-2:])
        out=self.head(q3)
        out.update(self.seg(q3,q4,q5,x.shape[-2:]))
        out["depth_inverse"]=depth
        return out

# ----------------- multi-layer image augmentation -----------------

class MultiLayerAugment(nn.Module):
    """
    Differentiable multi-view augmentation for training.
    Produces several differently transformed views of ONE image and returns
    them stacked as a batch. This is intended for consistency training.
    """
    def __init__(self,n_views=4):
        super().__init__()
        self.n_views=n_views
    def forward(self,x):
        views=[x]
        for i in range(self.n_views-1):
            v=x
            if i%2==0: v=torch.flip(v,[-1])
            # scale contrast around per-image mean
            mean=v.mean(dim=(-2,-1),keepdim=True)
            factor=0.85+0.1*(i+1)
            v=(v-mean)*factor+mean
            # small differentiable blur for one branch
            if i%3==1:
                v=F.avg_pool2d(v,3,1,1)
            views.append(v.clamp(0,1))
        return torch.cat(views,0)

def consistency_loss(pred_a,pred_b):
    """Generic feature consistency loss; use on corresponding augmented views."""
    return F.smooth_l1_loss(pred_a,pred_b.detach())
