"""
Structural Pruning and Graph Reconstruction Subsystem:
Replaces weight-masking with true channel/filter pruning and parameter reconstruction.
Physically reduces parameter count, tensor dimensions, memory footprint, and FLOPs.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class StructuralGraphPruner:
    """
    True Structural Channel Pruner with Graph Tensor Reconstruction:
    - Analyzes L1-norm across output channels of Conv2d layers.
    - Slices weight and bias tensors to eliminate low-salience filters.
    - Slices corresponding BatchNorm2d running statistics (mean, variance, weight, bias).
    - Slices input channels of immediately downstream Convolutions.
    - Physically reduces parameter count and tensor operations.
    """
    def __init__(self, prune_ratio: float = 0.25, min_channels: int = 8):
        self.prune_ratio = prune_ratio
        self.min_channels = min_channels

    def prune_conv_bn_pair(
        self,
        conv: nn.Conv2d,
        bn: Optional[nn.BatchNorm2d],
        next_conv: Optional[nn.Conv2d] = None,
        prune_ratio: Optional[float] = None
    ) -> Tuple[nn.Conv2d, Optional[nn.BatchNorm2d], Optional[nn.Conv2d], int]:
        """
        Prunes output channels of conv + bn, and matching input channels of next_conv.
        Returns reconstructed layers and count of channels pruned.
        """
        ratio = prune_ratio or self.prune_ratio
        out_channels = conv.out_channels
        num_to_keep = max(self.min_channels, int(out_channels * (1.0 - ratio)))
        num_pruned = out_channels - num_to_keep

        if num_pruned <= 0 or num_to_keep >= out_channels:
            return conv, bn, next_conv, 0

        # Calculate L1 norms of filters
        with torch.no_grad():
            norms = conv.weight.data.view(out_channels, -1).abs().sum(dim=1)
            _, keep_indices = torch.topk(norms, num_to_keep, largest=True)
            keep_indices = torch.sort(keep_indices)[0]

        # Reconstruct conv with sliced output channels
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=num_to_keep,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups if conv.groups == 1 else 1,
            bias=conv.bias is not None
        ).to(conv.weight.device)

        new_conv.weight.data = conv.weight.data[keep_indices, :, :, :].clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_indices].clone()

        # Reconstruct BN if present
        new_bn = None
        if bn is not None:
            new_bn = nn.BatchNorm2d(
                num_features=num_to_keep,
                eps=bn.eps,
                momentum=bn.momentum,
                affine=bn.affine,
                track_running_stats=bn.track_running_stats
            ).to(bn.weight.device)
            if bn.affine:
                new_bn.weight.data = bn.weight.data[keep_indices].clone()
                new_bn.bias.data = bn.bias.data[keep_indices].clone()
            if bn.track_running_stats:
                new_bn.running_mean.data = bn.running_mean.data[keep_indices].clone()
                new_bn.running_var.data = bn.running_var.data[keep_indices].clone()

        # Reconstruct next_conv if present (slice in_channels)
        new_next_conv = None
        if next_conv is not None and next_conv.groups == 1:
            new_next_conv = nn.Conv2d(
                in_channels=num_to_keep,
                out_channels=next_conv.out_channels,
                kernel_size=next_conv.kernel_size,
                stride=next_conv.stride,
                padding=next_conv.padding,
                dilation=next_conv.dilation,
                groups=1,
                bias=next_conv.bias is not None
            ).to(next_conv.weight.device)
            new_next_conv.weight.data = next_conv.weight.data[:, keep_indices, :, :].clone()
            if next_conv.bias is not None:
                new_next_conv.bias.data = next_conv.bias.data.clone()
        else:
            new_next_conv = next_conv

        return new_conv, new_bn, new_next_conv, num_pruned

    def prune_model_stem_and_heads(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, any]]:
        """
        Applies structural filter pruning to internal modules of the model
        and returns the reconstructed network with verified physical parameter reduction.
        """
        pruned_model = copy.deepcopy(model)
        orig_params = sum(p.numel() for p in model.parameters())
        pruned_layers_info = []

        # Target classification and regression convolution branches
        if hasattr(pruned_model, "det_head"):
            det_head = pruned_model.det_head
            # Prune internal intermediate convolutions in cls_convs and reg_convs
            for i in range(len(det_head.cls_convs)):
                # Each is a Sequential of ConvBNAct
                seq = det_head.cls_convs[i]
                if len(seq) >= 2:
                    c1_block = seq[0]
                    c2_block = seq[1]
                    if hasattr(c1_block, "conv") and hasattr(c1_block, "bn") and hasattr(c2_block, "conv"):
                        new_c1, new_bn, new_c2, pruned_cnt = self.prune_conv_bn_pair(
                            c1_block.conv, c1_block.bn, c2_block.conv
                        )
                        if pruned_cnt > 0:
                            c1_block.conv = new_c1
                            c1_block.bn = new_bn
                            c2_block.conv = new_c2
                            pruned_layers_info.append(f"det_head.cls_convs[{i}]: -{pruned_cnt} channels")

            for i in range(len(det_head.reg_convs)):
                seq = det_head.reg_convs[i]
                if len(seq) >= 2:
                    c1_block = seq[0]
                    c2_block = seq[1]
                    if hasattr(c1_block, "conv") and hasattr(c1_block, "bn") and hasattr(c2_block, "conv"):
                        new_c1, new_bn, new_c2, pruned_cnt = self.prune_conv_bn_pair(
                            c1_block.conv, c1_block.bn, c2_block.conv
                        )
                        if pruned_cnt > 0:
                            c1_block.conv = new_c1
                            c1_block.bn = new_bn
                            c2_block.conv = new_c2
                            pruned_layers_info.append(f"det_head.reg_convs[{i}]: -{pruned_cnt} channels")

        new_params = sum(p.numel() for p in pruned_model.parameters())
        reduction_ratio = (orig_params - new_params) / max(1, orig_params)

        stats = {
            "original_parameters": orig_params,
            "pruned_parameters": new_params,
            "parameter_reduction_percent": reduction_ratio * 100.0,
            "pruned_layers": pruned_layers_info,
            "structurally_reduced": new_params < orig_params
        }
        return pruned_model, stats
