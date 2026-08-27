from __future__ import annotations

from typing import Optional

import numpy as np

from src.preprocessing.roi.roi_extract import BBox, crop_ellipse_region, detect_compact_face_bbox
from .base import BaseROITransformStrategy


class EllipseROITransformStrategy(BaseROITransformStrategy):
    name = 'ellipse'

    def detect_bbox(self, frames, pad=0.0, forehead_ratio=0.20, side_ratio=0.20, bottom_ratio=0.20, max_frames=30):
        return detect_compact_face_bbox(frames, pad=pad, forehead_ratio=forehead_ratio, side_ratio=side_ratio, bottom_ratio=bottom_ratio, max_frames=max_frames)

    def apply(self, frame: np.ndarray, bbox: Optional[BBox], size: int = 72, features=None) -> np.ndarray:
        return crop_ellipse_region(frame, bbox=bbox, size=size)
