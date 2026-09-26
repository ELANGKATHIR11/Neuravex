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

class RepLKConv(nn.Module):
    """
    Reparameterizable Large-Kernel Conv (RepLK 7x7 + 3x3 + 1x1):
    Expands effective receptive field to ViT scale without quadratic memory cost.
    Training: 7x7 Depthwise Conv + 3x3 Depthwise Conv + 1x1 Conv
    Deploy: Fused into a single equivalent 7x7 Depthwise Conv2d.
    """
    def __init__(self, c, k=7):
        super().__init__()
        self.c = c
        self.k = k
        pad = k // 2
        self.conv_large = nn.Conv2d(c, c, k, 1, pad, groups=c, bias=False)
        self.bn_large = nn.BatchNorm2d(c)
        self.conv_small = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.bn_small = nn.BatchNorm2d(c)
        self.fused_conv = None

    def forward(self, x):
        if self.fused_conv is not None:
            return self.fused_conv(x)
        return self.bn_large(self.conv_large(x)) + self.bn_small(self.conv_small(x))

    def switch_to_deploy(self):
        if self.fused_conv is not None:
            return
        # Fuse BNs
        std_l = (self.bn_large.running_var + self.bn_large.eps).sqrt()
        w_l = self.conv_large.weight * (self.bn_large.weight / std_l).reshape(-1, 1, 1, 1)
        b_l = self.bn_large.bias - self.bn_large.running_mean * self.bn_large.weight / std_l

        std_s = (self.bn_small.running_var + self.bn_small.eps).sqrt()
        w_s = self.conv_small.weight * (self.bn_small.weight / std_s).reshape(-1, 1, 1, 1)
        b_s = self.bn_small.bias - self.bn_small.running_mean * self.bn_small.weight / std_s

        pad_diff = (self.k - 3) // 2
        w_s_padded = F.pad(w_s, [pad_diff, pad_diff, pad_diff, pad_diff])

        w_fused = w_l + w_s_padded
        b_fused = b_l + b_s

        self.fused_conv = nn.Conv2d(self.c, self.c, self.k, 1, self.k // 2, groups=self.c, bias=True)
        self.fused_conv.weight.data = w_fused
        self.fused_conv.bias.data = b_fused
        del self.conv_large, self.bn_large, self.conv_small, self.bn_small

class PartialChannelRepBlock(nn.Module):
    """
    Partial Channel RepBlock (P-RepBlock) with RepLK Large-Kernel 7x7 Depthwise Conv:
    Applies reparameterizable 7x7 large-kernel convolutions and depthwise SE attention
    to a partial slice of channels while identity-passing the remainder.
    Significantly cuts FLOPs and latency while matching ViT receptive field!
    """
    def __init__(self, c, part_ratio=0.5, e=1.0, use_replk=True):
        super().__init__()
        self.c = c
        self.part_c = int(c * part_ratio)
        self.pass_c = c - self.part_c
        h = int(self.part_c * e)

        self.cv1 = RepConv(self.part_c, h, s=1)
        if use_replk:
            self.dw = RepLKConv(h, k=7)
        else:
            self.dw = nn.Sequential(
                nn.Conv2d(h, h, 3, 1, 1, groups=h, bias=False),
                nn.BatchNorm2d(h)
            )
        self.se = SqueezeExcitation(h)
        self.cv2 = ConvBNAct(h, self.part_c, 1)

    def forward(self, x):
        x_part, x_pass = torch.split(x, [self.part_c, self.pass_c], dim=1)
        x_out = self.cv2(self.se(F.silu(self.dw(self.cv1(x_part)))))
        out = torch.cat([x_part + x_out, x_pass], dim=1)
        return out

    def switch_to_deploy(self):
        if hasattr(self.cv1, "switch_to_deploy"):
            self.cv1.switch_to_deploy()
        if hasattr(self.dw, "switch_to_deploy"):
            self.dw.switch_to_deploy()


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

    def switch_to_deploy(self):
        for m in self.modules():
            if m is not self and hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()


class MaskedMultimodalAutoencoder(nn.Module):
    """
    Masked Multimodal Autoencoding (MMA-Neuravex):
    Cross-modal self-supervised pre-training:
      - Randomly masks a percentage of RGB image patches (e.g., 60%) and Depth tokens (e.g., 75%)
      - Cross-modal transformer / bottleneck reconstructs missing visual patches from available depth
        and missing depth tokens from available visual tokens.
      - Forces representations to discover that visual texture implies 3D surface geometry,
        and depth outlines indicate physical boundaries.
    """
    def __init__(self, backbone: Backbone, patch_size: int = 16, embed_dim: int = 256,
                 rgb_mask_ratio: float = 0.60, depth_mask_ratio: float = 0.75):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.rgb_mask_ratio = rgb_mask_ratio
        self.depth_mask_ratio = depth_mask_ratio

        # Depth token projection
        self.depth_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Visual patch projection
        self.rgb_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Cross-modal fusion bottleneck
        self.cross_fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Decoders
        self.rgb_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3 * patch_size * patch_size)
        )
        self.depth_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1 * patch_size * patch_size)
        )

    def patchify(self, imgs: torch.Tensor, channels: int) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, C * patch_size * patch_size)"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h_p, w_p = H // p, W // p
        x = imgs.reshape(B, C, h_p, p, w_p, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        patches = x.reshape(B, h_p * w_p, C * p * p)
        return patches

    def generate_random_mask(self, batch_size: int, num_patches: int, mask_ratio: float, device: torch.device):
        """Generates boolean mask where True indicates MASKED tokens."""
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :len_keep], False)
        return mask

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> dict:
        """
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W) metric depth map
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        B, _, H, W = rgb.shape
        device = rgb.device

        # 1. Project to patch tokens
        rgb_tokens = self.rgb_proj(rgb).flatten(2).transpose(1, 2) # (B, N, D)
        depth_tokens = self.depth_proj(depth).flatten(2).transpose(1, 2) # (B, N, D)
        num_patches = rgb_tokens.shape[1]

        # 2. Random masking
        rgb_mask = self.generate_random_mask(B, num_patches, self.rgb_mask_ratio, device)
        depth_mask = self.generate_random_mask(B, num_patches, self.depth_mask_ratio, device)

        # Zero-out masked tokens
        rgb_visible = rgb_tokens * (~rgb_mask.unsqueeze(-1)).float()
        depth_visible = depth_tokens * (~depth_mask.unsqueeze(-1)).float()

        # 3. Cross-modal conditioning: reconstruct RGB from depth + visible RGB, and depth from RGB + visible depth
        fused = self.cross_fuse(torch.cat([rgb_visible, depth_visible], dim=-1))

        pred_rgb_patches = self.rgb_decoder(fused)
        pred_depth_patches = self.depth_decoder(fused)

        # 4. Compute reconstruction losses only on the masked tokens
        target_rgb_patches = self.patchify(rgb, 3)
        target_depth_patches = self.patchify(depth, 1)

        loss_rgb = ((pred_rgb_patches - target_rgb_patches) ** 2).mean(dim=-1)
        loss_rgb = (loss_rgb * rgb_mask.float()).sum() / (rgb_mask.sum().clamp_min(1.0))

        loss_depth = ((pred_depth_patches - target_depth_patches) ** 2).mean(dim=-1)
        loss_depth = (loss_depth * depth_mask.float()).sum() / (depth_mask.sum().clamp_min(1.0))

        total_loss = loss_rgb + 0.5 * loss_depth

        return {
            "loss_mma": total_loss,
            "loss_rgb_recon": loss_rgb.detach(),
            "loss_depth_recon": loss_depth.detach(),
            "rgb_mask": rgb_mask,
            "depth_mask": depth_mask
        }

