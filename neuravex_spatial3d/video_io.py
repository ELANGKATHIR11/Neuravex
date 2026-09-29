"""
Native Video Stream & File Processor:
Frame-by-frame decoding, FPS pacing, stream generator, and video writer.
Supports MP4, AVI, MKV, RTSP/HTTP streams, and USB/Webcam devices.
"""

import os
from typing import Generator, Tuple, Optional, Union
import cv2
import numpy as np

class VideoStreamProcessor:
    """
    Decodes video files or live camera streams into sequential frames for 3D spatial analysis.
    """
    def __init__(self, source: Union[str, int]):
        """
        Args:
            source: Video file path (str), RTSP/HTTP stream URL (str), or device index (int e.g. 0).
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (frame_idx, rgb_frame) sequentially.
        """
        idx = 0
        while self.cap.isOpened():
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield idx, rgb
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break

    def release(self):
        """Release capture device."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """
    Encodes annotated RGB frames into an output MP4/AVI video file.
    """
    def __init__(self, output_path: str, fps: float = 30.0, frame_size: Tuple[int, int] = (1920, 1080)):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write_frame(self, rgb_frame: np.ndarray):
        """Writes an RGB image frame to video."""
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
