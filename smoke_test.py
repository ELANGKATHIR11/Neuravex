
import sys, torch
from yolo27_v0_3 import YOLO27
from multitask_training import AdaptiveTaskWeights, boxes3d_corners, iou3d_aabb
from augmentation import SingleImageMultiViewAugment

m=YOLO27()
x=torch.rand(1,3,320,320)
with torch.inference_mode(): y=m(x)
print("parameters:",sum(p.numel() for p in m.parameters()))
print("outputs:",{k:(list(v.shape) if hasattr(v,"shape") else "parameter") for k,v in y.items()})
v=SingleImageMultiViewAugment(4)(x)
print("augmented:",list(v.shape))
c=boxes3d_corners(torch.tensor([[0.,0.,5.]]),torch.tensor([[4.,2.,1.5]]),torch.tensor([0.]))
print("self 3D IoU:",iou3d_aabb(c,c).item())
w=AdaptiveTaskWeights(["det","semantic","geometry"])
losses={k:torch.ones(()) for k in ["det","semantic","geometry"]}
print("adaptive loss:",w(losses)[0].item())
