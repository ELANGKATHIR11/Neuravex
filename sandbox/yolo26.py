"""
YOLO26 Reference Model for Sandbox Benchmarking.
Implements the 2026 anchor-free detection baseline:
- C2f/RepNCSPELAN4-style residual backbone (P3, P4, P5)
- Standard decoupled detection neck (PANet)
- Anchor-free 2D detection head (cls + reg with DFL)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, 3, 1)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, shortcut) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLO26Backbone(nn.Module):
    def __init__(self, base_c=48):
        super().__init__()
        b = base_c
        self.stem = nn.Sequential(Conv(3, b, 3, 2), Conv(b, b * 2, 3, 2))
        self.stage2 = nn.Sequential(C2f(b * 2, b * 2, 2), Conv(b * 2, b * 4, 3, 2))
        self.stage3 = nn.Sequential(C2f(b * 4, b * 4, 4), Conv(b * 4, b * 8, 3, 2))
        self.stage4 = nn.Sequential(C2f(b * 8, b * 8, 4), Conv(b * 8, b * 16, 3, 2))
        self.stage5 = nn.Sequential(C2f(b * 16, b * 16, 2), SPPF(b * 16, b * 16))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = x  # stride 8
        x = self.stage3(x)
        p4 = x  # stride 16
        x = self.stage4(x)
        p5 = self.stage5(x)  # stride 32
        return p3, p4, p5

class YOLO26(nn.Module):
    """
    Standard YOLO26 Anchor-Free 2D Detector (comparative baseline).
    """
    def __init__(self, num_classes=80, base_c=48):
        super().__init__()
        self.backbone = YOLO26Backbone(base_c)
        c3, c4, c5 = base_c * 4, base_c * 8, base_c * 16
        out_c = base_c * 4

        # FPN / PANet
        self.lat5 = Conv(c5, out_c, 1)
        self.lat4 = Conv(c4, out_c, 1)
        self.top_p4 = Conv(out_c * 2, out_c, 3)
        self.top_p3 = Conv(out_c + c3, out_c, 3)

        self.down_p3 = Conv(out_c, out_c, 3, 2)
        self.bot_p4 = Conv(out_c * 2, out_c, 3)
        self.down_p4 = Conv(out_c, out_c, 3, 2)
        self.bot_p5 = Conv(out_c * 2, out_c, 3)

        # Standard 2D decoupled heads (cls + box)
        self.cls_heads = nn.ModuleList([nn.Conv2d(out_c, num_classes, 1) for _ in range(3)])
        self.box_heads = nn.ModuleList([nn.Conv2d(out_c, 4, 1) for _ in range(3)])

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        
        # Top-down
        u5 = F.interpolate(self.lat5(p5), size=p4.shape[-2:], mode="nearest")
        t4 = self.top_p4(torch.cat([self.lat4(p4), u5], 1))
        u4 = F.interpolate(t4, size=p3.shape[-2:], mode="nearest")
        q3 = self.top_p3(torch.cat([p3, u4], 1))

        # Bottom-up
        d3 = self.down_p3(q3)
        q4 = self.bot_p4(torch.cat([t4, d3], 1))
        d4 = self.down_p4(q4)
        q5 = self.bot_p5(torch.cat([self.lat5(p5), d4], 1))

        feats = [q3, q4, q5]
        cls_preds = [h(f).flatten(2).permute(0, 2, 1) for h, f in zip(self.cls_heads, feats)]
        box_preds = [h(f).flatten(2).permute(0, 2, 1) for h, f in zip(self.box_heads, feats)]

        return {
            "class_logits": torch.cat(cls_preds, dim=1),
            "pred_boxes": torch.cat(box_preds, dim=1)
        }
