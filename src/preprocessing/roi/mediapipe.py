from __future__ import annotations

from typing import Optional

import numpy as np

from src.preprocessing.roi.roi_extract import BBox, crop_bbox_region, detect_face_bbox_mediapipe
from .base import BaseROITransformStrategy


class MediaPipeROITransformStrategy(BaseROITransformStrategy):
    name = 'mediapipe'

    def detect_bbox(self, frames, pad=0.0, forehead_ratio=0.20, side_ratio=0.20, bottom_ratio=0.20, max_frames=30):
        return detect_face_bbox_mediapipe(frames, pad=pad, max_frames=max_frames)

    def apply(self, frame: np.ndarray, bbox: Optional[BBox], size: int = 72, features=None) -> np.ndarray:
        return crop_bbox_region(frame, bbox=bbox, size=size)
