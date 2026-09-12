
import torch
import torch.nn.functional as F

class SingleImageMultiViewAugment:
    """
    Generates N deterministic/differentiable views from one image.
    Geometry metadata must be transformed together with each view.
    """
    def __init__(self,n_views=4):
        self.n_views=n_views
    def __call__(self,x):
        views=[x]
        for i in range(1,self.n_views):
            v=x
            if i%2: v=torch.flip(v,[-1])
            mean=v.mean((-2,-1),keepdim=True)
            gain=0.85+0.10*(i%3)
            v=(v-mean)*gain+mean
            if i%3==2: v=F.avg_pool2d(v,3,1,1)
            views.append(v.clamp(0,1))
        return torch.cat(views,0)
