from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

from src.preprocessing.roi.roi_extract import BBox

logger = logging.getLogger(__name__)


class BaseROITransformStrategy:
    """Base interface for registered ROI strategies."""

    name = 'base'

    def detect_bbox(
        self,
        frames: Sequence[np.ndarray],
        pad: float = 0.0,
        forehead_ratio: float = 0.20,
        side_ratio: float = 0.20,
        bottom_ratio: float = 0.20,
        max_frames: int = 30,
    ) -> Optional[BBox]:
        return None

    def apply(
        self,
        frame: np.ndarray,
        bbox: Optional[BBox],
        size: int = 72,
        features: Optional[dict[str, tuple[float, float] | tuple[float, float, float, float] | list[tuple[float, float]]]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    # New strategies can implement detect_bbox/apply/transform directly without
    # adding a new branch in roi_extract.py.

    def transform(
        self,
        frames: np.ndarray,
        size: int = 72,
        pad: float = 0.0,
        forehead_ratio: float = 0.20,
        side_ratio: float = 0.20,
        bottom_ratio: float = 0.20,
        align_nose_axis: bool = False,
        face_mesh_first_frame_mask: bool = False,
        center_face: bool = False,
        canonical_face_mask: bool = False,
        frame_independent_face_mesh: bool = False,
    ) -> tuple[np.ndarray, Optional[BBox]]:
        bbox = self.detect_bbox(
            frames,
            pad=pad,
            forehead_ratio=forehead_ratio,
            side_ratio=side_ratio,
            bottom_ratio=bottom_ratio,
        )
        processed: list[np.ndarray] = []
        for frame in frames:
            processed.append(self.apply(frame, bbox=bbox, size=size))
        return np.stack(processed, axis=0), bbox
