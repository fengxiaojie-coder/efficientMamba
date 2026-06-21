"""Compatibility wrappers for compact ROI cropping.

This module keeps the public ROI helpers available while defaulting to the
compact bbox crop path used by the preprocessing pipeline.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .roi_extract import (
    BBox,
    apply_roi_and_resize,
    detect_face_bbox as _detect_face_bbox,
    extract_roi as _extract_roi,
    get_face_bbox as _get_face_bbox,
    roi_extract as _roi_extract,
    transform_frames_with_roi as _transform_frames_with_roi,
)

DEFAULT_ROI = "bbox"


def detect_face_bbox_haar(frames: Sequence[np.ndarray], pad: float = 0.0, max_frames: int = 30) -> Optional[BBox]:
    return _detect_face_bbox(frames, roi="haar", pad=pad, max_frames=max_frames)


def detect_face_bbox_mediapipe(frames: Sequence[np.ndarray], pad: float = 0.0, max_frames: int = 30) -> Optional[BBox]:
    return _detect_face_bbox(frames, roi="mediapipe", pad=pad, max_frames=max_frames)


def detect_face_bbox(frames: Sequence[np.ndarray], roi: str = DEFAULT_ROI, pad: float = 0.0, max_frames: int = 30) -> Optional[BBox]:
    return _detect_face_bbox(frames, roi=roi, pad=pad, max_frames=max_frames)


def extract_roi(frame: np.ndarray, roi: str = DEFAULT_ROI, size: int = 72, bbox: Optional[BBox] = None) -> np.ndarray:
    return _extract_roi(frame, roi=roi, size=size, bbox=bbox)


def transform_frames_with_roi(
    frames: np.ndarray,
    roi: str = DEFAULT_ROI,
    size: int = 72,
    pad: float = 0.0,
) -> tuple[np.ndarray, Optional[BBox]]:
    return _transform_frames_with_roi(frames, roi=roi, size=size, pad=pad)


def get_face_bbox(*args, **kwargs):
    return _get_face_bbox(*args, **kwargs)


def roi_extract(*args, **kwargs):
    return _roi_extract(*args, **kwargs)


def crop_and_resize_roi(frame: np.ndarray, roi: str = DEFAULT_ROI, size: int = 72, bbox: Optional[BBox] = None) -> np.ndarray:
    return apply_roi_and_resize(frame, roi=roi, size=size, bbox=bbox)


__all__ = [
    "BBox",
    "DEFAULT_ROI",
    "apply_roi_and_resize",
    "crop_and_resize_roi",
    "detect_face_bbox",
    "detect_face_bbox_haar",
    "detect_face_bbox_mediapipe",
    "extract_roi",
    "get_face_bbox",
    "roi_extract",
    "transform_frames_with_roi",
]