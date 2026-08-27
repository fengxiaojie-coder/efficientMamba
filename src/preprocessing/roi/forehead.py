from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.preprocessing.roi.roi_extract import BBox, detect_compact_face_bbox
from .base import BaseROITransformStrategy


class ForeheadROITransformStrategy(BaseROITransformStrategy):
    name = 'forehead'

    def detect_bbox(self, frames, pad=0.0, forehead_ratio=0.20, side_ratio=0.20, bottom_ratio=0.20, max_frames=30):
        # Reuse the compact face detector to get a stable face box first.
        return detect_compact_face_bbox(
            frames,
            pad=pad,
            forehead_ratio=forehead_ratio,
            side_ratio=side_ratio,
            bottom_ratio=bottom_ratio,
            max_frames=max_frames,
        )

    def apply(self, frame: np.ndarray, bbox: Optional[BBox], size: int = 72, features=None) -> np.ndarray:
        h, w = frame.shape[:2]
        if bbox is None:
            return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(y1 + 1, min(h, int(y2)))

        face_h = y2 - y1
        band_h = max(1, int(round(face_h * 0.35)))
        fh_y1 = y1
        fh_y2 = min(y2, y1 + band_h)

        forehead = frame[fh_y1:fh_y2, x1:x2]
        if forehead.size == 0:
            forehead = frame[y1:y2, x1:x2]
        return cv2.resize(forehead, (size, size), interpolation=cv2.INTER_AREA)
