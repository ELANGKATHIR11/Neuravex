"""
Native Image, Photo, and Multi-Scale Frame Loader and Preprocessor.
Supports numpy arrays, PyTorch tensors, OpenCV frames, PIL Images, and raw files.
"""

import os
from typing import Union, Tuple, Optional
import torch
import numpy as np
import cv2
from PIL import Image

class ImageProcessor:
    """
    Standardizes loading and normalization of photos, images, and camera frames.
    """
    @staticmethod
    def load_image(
        image_input: Union[str, np.ndarray, Image.Image, torch.Tensor],
        target_size: Optional[Tuple[int, int]] = (640, 640),
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Loads an image input from file path, numpy array, PIL Image, or torch Tensor.
        
        Returns:
            tensor_chw: (1, 3, H, W) normalized tensor in [0, 1] on device
            original_rgb: (H_orig, W_orig, 3) uint8 numpy image
            orig_hw: (H_orig, W_orig)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Failed decoding image: {image_input}")
            orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                orig_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                orig_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                orig_rgb = image_input.copy()
        elif isinstance(image_input, Image.Image):
            orig_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                image_input = image_input.squeeze(0)
            if image_input.shape[0] == 3:
                orig_rgb = (image_input.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            else:
                orig_rgb = image_input.cpu().numpy().astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        h_orig, w_orig = orig_rgb.shape[:2]

        # Resize for neural network input if requested
        if target_size is not None:
            w_tgt, h_tgt = target_size
            resized = cv2.resize(orig_rgb, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        else:
            resized = orig_rgb

        # Normalize to float32 [0.0, 1.0] and permute to (1, 3, H, W)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor.to(device), orig_rgb, (h_orig, w_orig)
