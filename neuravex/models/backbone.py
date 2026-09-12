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

class RepConv(nn.Module):
    """
    Reparameterizable 3x3 Conv:
    Training: 3x3 conv + 1x1 conv + identity residual
    Inference: Fused into a single equivalent 3x3 Conv for zero latency/memory overhead!
    """
    def __init__(self, c1, c2, s=1, p=1, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.g = g
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        # Training multi-branch
        self.conv3x3 = nn.Conv2d(c1, c2, 3, s, p, groups=g, bias=False)
        self.bn3x3 = nn.BatchNorm2d(c2)

        self.conv1x1 = nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)
        self.bn1x1 = nn.BatchNorm2d(c2)

        self.identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

        # Fused inference conv (None until fuse_repconv() is called)
        self.fused_conv = None

    def forward(self, x):
        if self.fused_conv is not None:
            return self.act(self.fused_conv(x))

        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.identity is not None:
            out += self.identity(x)
        return self.act(out)

    def switch_to_deploy(self):
        """Fuses multi-branch into a single 3x3 Conv2d."""
        if self.fused_conv is not None:
            return
        w3, b3 = self._fuse_bn(self.conv3x3, self.bn3x3)
        w1, b1 = self._fuse_bn(self.conv1x1, self.bn1x1)
        w1_padded = F.pad(w1, [1, 1, 1, 1])

        w_fused = w3 + w1_padded
        b_fused = b3 + b1

        if self.identity is not None:
            w_id = torch.zeros_like(w3)
            for i in range(self.c1):
                w_id[i, i % (self.c1 // self.g), 1, 1] = 1.0
            id_bn = self.identity
            std = (id_bn.running_var + id_bn.eps).sqrt()
            w_id = w_id * (id_bn.weight / std).reshape(-1, 1, 1, 1)
            b_id = id_bn.bias - id_bn.running_mean * id_bn.weight / std
            w_fused += w_id
            b_fused += b_id

        self.fused_conv = nn.Conv2d(self.c1, self.c2, 3, self.s, 1, groups=self.g, bias=True)
        self.fused_conv.weight.data = w_fused
        self.fused_conv.bias.data = b_fused

        # Remove training branches to free VRAM
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1, self.identity

    def _fuse_bn(self, conv, bn):
        w = conv.weight
        std = (bn.running_var + bn.eps).sqrt()
        w_fused = w * (bn.weight / std).reshape(-1, 1, 1, 1)
        b_fused = bn.bias - bn.running_mean * bn.weight / std
        return w_fused, b_fused

class PartialChannelRepBlock(nn.Module):
    """
    Partial Channel RepBlock (P-RepBlock):
    Applies reparameterizable convolutions and depthwise SE attention
    to a partial slice of channels while identity-passing the remainder.
    Significantly cuts FLOPs and latency while preserving rich feature gradients!
    """
    def __init__(self, c, part_ratio=0.5, e=1.0):
        super().__init__()
        self.c = c
        self.part_c = int(c * part_ratio)
        self.pass_c = c - self.part_c
        h = int(self.part_c * e)

        self.cv1 = RepConv(self.part_c, h, s=1)
        self.dw = nn.Conv2d(h, h, 3, 1, 1, groups=h, bias=False)
        self.bn_dw = nn.BatchNorm2d(h)
        self.se = SqueezeExcitation(h)
        self.cv2 = ConvBNAct(h, self.part_c, 1)

    def forward(self, x):
        x_part, x_pass = torch.split(x, [self.part_c, self.pass_c], dim=1)
        x_out = self.cv2(self.se(F.silu(self.bn_dw(self.dw(self.cv1(x_part))))))
        out = torch.cat([x_part + x_out, x_pass], dim=1)
        return out

RepBlock = PartialChannelRepBlock

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
