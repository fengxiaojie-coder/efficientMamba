"""Registration helpers for ROI transform strategies."""
from __future__ import annotations

from src.preprocessing.registry import register_roi_strategy
from src.preprocessing.roi import (
    BBoxROITransformStrategy,
    EllipseROITransformStrategy,
    FaceMeshROITransformStrategy,
    ForeheadROITransformStrategy,
    FullROITransformStrategy,
    HaarROITransformStrategy,
    MediaPipeROITransformStrategy,
)


def register_default_roi_strategies() -> None:
    """Register the built-in ROI strategies used by the preprocessing pipeline."""
    register_roi_strategy('full', 'Full frame ROI strategy', lambda: FullROITransformStrategy())
    register_roi_strategy('bbox', 'Compact bounding-box ROI strategy', lambda: BBoxROITransformStrategy())
    register_roi_strategy('haar', 'Haar cascade ROI strategy', lambda: HaarROITransformStrategy())
    register_roi_strategy('mediapipe', 'MediaPipe bbox ROI strategy', lambda: MediaPipeROITransformStrategy())
    register_roi_strategy('ellipse', 'Ellipse mask ROI strategy', lambda: EllipseROITransformStrategy())
    register_roi_strategy('face_mesh', 'Face mesh ROI strategy', lambda: FaceMeshROITransformStrategy())
    register_roi_strategy('forehead', 'Forehead-band ROI strategy', lambda: ForeheadROITransformStrategy())
