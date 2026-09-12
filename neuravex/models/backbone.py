import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, h, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class RepBlock(nn.Module):
    """Residual bottleneck block with depthwise separation and SE attention."""
    def __init__(self, c, e=1.0):
        super().__init__()
        h = int(c * e)
        self.cv1 = ConvBNAct(c, h, 1)
        self.cv2 = ConvBNAct(h, h, 3, 1, g=h)
        self.se = SqueezeExcitation(h)
        self.cv3 = nn.Sequential(
            nn.Conv2d(h, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        return F.silu(x + self.cv3(self.se(self.cv2(self.cv1(x)))))

class Backbone(nn.Module):
    """
    Hierarchical multi-scale backbone producing feature maps P3, P4, P5
    at strides 8, 16, 32 with functional depth_mul scaling.
    """
    def __init__(self, base_c=48, depth_mul=1.0):
        super().__init__()
        b = base_c
        self.depth_mul = depth_mul

        def num_blocks(n):
            return max(round(n * depth_mul), 1)

        # Stem: stride 2
        self.stem = nn.Sequential(
            ConvBNAct(3, b, 3, 2),
            ConvBNAct(b, b, 3, 1)
        )
        # Stage 2: stride 4
        self.s2 = nn.Sequential(
            ConvBNAct(b, b * 2, 3, 2),
            *[RepBlock(b * 2) for _ in range(num_blocks(2))]
        )
        # Stage 3 (P3): stride 8
        self.s3 = nn.Sequential(
            ConvBNAct(b * 2, b * 4, 3, 2),
            *[RepBlock(b * 4) for _ in range(num_blocks(3))]
        )
        # Stage 4 (P4): stride 16
        self.s4 = nn.Sequential(
            ConvBNAct(b * 4, b * 8, 3, 2),
            *[RepBlock(b * 8) for _ in range(num_blocks(3))]
        )
        # Stage 5 (P5): stride 32
        self.s5 = nn.Sequential(
            ConvBNAct(b * 8, b * 16, 3, 2),
            *[RepBlock(b * 16) for _ in range(num_blocks(2))]
        )

    def forward(self, x):
        x = self.stem(x)
        p2 = self.s2(x)
        p3 = self.s3(p2)
        p4 = self.s4(p3)
        p5 = self.s5(p4)
        return p3, p4, p5
