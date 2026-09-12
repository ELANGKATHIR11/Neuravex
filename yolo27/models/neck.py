import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct, RepBlock

class BidirectionalCrossTaskFusion(nn.Module):
    """
    Lightweight bidirectional cross-task feature fusion:
    F_i' = F_i + sum_{j != i} G_{ij}(F_i, F_j) * F_j
    """
    def __init__(self, channels: int):
        super().__init__()
        self.c = channels
        # Gating networks
        self.gate_det_from_seg = nn.Sequential(
            ConvBNAct(channels * 2, channels // 2, 1),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        self.gate_det_from_depth = nn.Sequential(
            ConvBNAct(channels * 2, channels // 2, 1),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        self.gate_seg_from_det = nn.Sequential(
            ConvBNAct(channels * 2, channels // 2, 1),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        self.gate_depth_from_det = nn.Sequential(
            ConvBNAct(channels * 2, channels // 2, 1),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, f_det, f_seg, f_depth):
        g_ds = self.gate_det_from_seg(torch.cat([f_det, f_seg], dim=1))
        g_dd = self.gate_det_from_depth(torch.cat([f_det, f_depth], dim=1))
        g_sd = self.gate_seg_from_det(torch.cat([f_seg, f_det], dim=1))
        g_dpd = self.gate_depth_from_det(torch.cat([f_depth, f_det], dim=1))

        f_det_fused = f_det + g_ds * f_seg + g_dd * f_depth
        f_seg_fused = f_seg + g_sd * f_det
        f_depth_fused = f_depth + g_dpd * f_det
        return f_det_fused, f_seg_fused, f_depth_fused

class PANetNeck(nn.Module):
    """
    Bidirectional Path Aggregation Network (PANet) fusing P3, P4, P5 into Q3, Q4, Q5.
    """
    def __init__(self, base_c=48):
        super().__init__()
        c3, c4, c5 = base_c * 4, base_c * 8, base_c * 16
        out_c = base_c * 4  # Common channel dimension

        self.reduce_p5 = ConvBNAct(c5, out_c, 1)
        self.reduce_p4 = ConvBNAct(c4, out_c, 1)
        self.reduce_p3 = ConvBNAct(c3, out_c, 1)

        # Top-down pathway
        self.top_down_p4 = ConvBNAct(out_c * 2, out_c, 3)
        self.top_down_p3 = ConvBNAct(out_c * 2, out_c, 3)

        # Bottom-up pathway
        self.down_p3_to_p4 = ConvBNAct(out_c, out_c, 3, 2)
        self.bottom_up_p4 = ConvBNAct(out_c * 2, out_c, 3)
        self.down_p4_to_p5 = ConvBNAct(out_c, out_c, 3, 2)
        self.bottom_up_p5 = ConvBNAct(out_c * 2, out_c, 3)

    def forward(self, p3, p4, p5):
        # Channel reduction
        r3 = self.reduce_p3(p3)
        r4 = self.reduce_p4(p4)
        r5 = self.reduce_p5(p5)

        # Top-down
        up5 = F.interpolate(r5, size=r4.shape[-2:], mode="nearest")
        td4 = self.top_down_p4(torch.cat([r4, up5], dim=1))

        up4 = F.interpolate(td4, size=r3.shape[-2:], mode="nearest")
        q3 = self.top_down_p3(torch.cat([r3, up4], dim=1))

        # Bottom-up
        down3 = self.down_p3_to_p4(q3)
        q4 = self.bottom_up_p4(torch.cat([td4, down3], dim=1))

        down4 = self.down_p4_to_p5(q4)
        q5 = self.bottom_up_p5(torch.cat([r5, down4], dim=1))

        return q3, q4, q5
