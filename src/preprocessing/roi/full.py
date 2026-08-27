from __future__ import annotations

from typing import Optional

import numpy as np

from src.preprocessing.roi.roi_extract import BBox, resize_full_frame
from .base import BaseROITransformStrategy


class FullROITransformStrategy(BaseROITransformStrategy):
    name = 'full'

    def apply(self, frame: np.ndarray, bbox: Optional[BBox], size: int = 72, features=None) -> np.ndarray:
        return resize_full_frame(frame, size=size)
