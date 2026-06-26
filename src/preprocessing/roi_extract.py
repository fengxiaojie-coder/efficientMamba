from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape [H, W, 3], got {frame.shape}")
    return frame


def _clip_bbox(x: int, y: int, w: int, h: int, width: int, height: int) -> BBox:
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _pad_bbox(bbox: BBox, pad: float, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    delta_x = int(round(box_w * pad))
    delta_y = int(round(box_h * pad))
    return _clip_bbox(x1 - delta_x, y1 - delta_y, box_w + 2 * delta_x, box_h + 2 * delta_y, width, height)


def _expand_bbox_directional(
    bbox: BBox,
    forehead_ratio: float,
    side_ratio: float,
    bottom_ratio: float,
    width: int,
    height: int,
) -> BBox:
    """Expand bbox with directional margins relative to face-box size.

    forehead_ratio: extra top margin relative to box height.
    side_ratio: extra left/right margin relative to box width.
    bottom_ratio: extra bottom margin relative to box height.
    """
    if forehead_ratio <= 0 and side_ratio <= 0 and bottom_ratio <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    extra_top = int(round(box_h * forehead_ratio))
    extra_side = int(round(box_w * side_ratio))
    extra_bottom = int(round(box_h * bottom_ratio))
    new_x1 = x1 - extra_side
    new_y1 = y1 - extra_top
    new_w = (x2 + extra_side) - new_x1
    new_h = (y2 + extra_bottom) - new_y1
    return _clip_bbox(new_x1, new_y1, new_w, new_h, width, height)


def _median_bbox(bboxes: Sequence[BBox]) -> Optional[BBox]:
    if not bboxes:
        return None
    arr = np.asarray(bboxes, dtype=np.float32)
    med = np.median(arr, axis=0).astype(int)
    return int(med[0]), int(med[1]), int(med[2]), int(med[3])


def detect_face_bbox_haar(
    frames: Sequence[np.ndarray],
    pad: float = 0.0,
    forehead_ratio: float = 0.20,
    side_ratio: float = 0.20,
    bottom_ratio: float = 0.20,
    max_frames: int = 30,
) -> Optional[BBox]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logger.warning("Haar cascade not available at %s", cascade_path)
        return None

    detected: list[BBox] = []
    for frame in list(frames)[:max_frames]:
        frame = _ensure_rgb(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        detected.append((int(x), int(y), int(x + w), int(y + h)))

    bbox = _median_bbox(detected)
    if bbox is None:
        return None
    h, w = frames[0].shape[:2]
    padded = _pad_bbox(bbox, pad, w, h)
    return _expand_bbox_directional(
        padded,
        forehead_ratio=forehead_ratio,
        side_ratio=side_ratio,
        bottom_ratio=bottom_ratio,
        width=w,
        height=h,
    )


def detect_face_bbox_mediapipe(frames: Sequence[np.ndarray], pad: float = 0.0, max_frames: int = 30) -> Optional[BBox]:
    # Import lazily so non-mediapipe ROI modes (e.g., haar/full) do not depend on
    # mediapipe binary compatibility in the runtime environment.
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning("mediapipe is not installed; falling back to full-frame ROI")
        return None

    detected: list[BBox] = []
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    try:
        for frame in list(frames)[:max_frames]:
            frame = _ensure_rgb(frame)
            result = face_detection.process(frame)
            if not result.detections:
                continue

            height, width = frame.shape[:2]
            for detection in result.detections:
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * width)
                y = int(box.ymin * height)
                w_box = int(box.width * width)
                h_box = int(box.height * height)
                detected.append((x, y, x + w_box, y + h_box))

        bbox = _median_bbox(detected)
        if bbox is None:
            return None
        height, width = frames[0].shape[:2]
        return _pad_bbox(bbox, pad, width, height)
    finally:
        face_detection.close()


def _bbox_to_slices(bbox: Optional[BBox], height: int, width: int) -> Tuple[slice, slice]:
    if bbox is None:
        return slice(0, height), slice(0, width)
    x1, y1, x2, y2 = bbox
    return slice(max(0, y1), min(height, y2)), slice(max(0, x1), min(width, x2))


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def _default_face_bbox(frame: np.ndarray) -> BBox:
    height, width = frame.shape[:2]
    x1 = int(width * 0.18)
    y1 = int(height * 0.10)
    x2 = int(width * 0.82)
    y2 = int(height * 0.92)
    return x1, y1, x2, y2


def _apply_ellipse_mask(frame: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    frame = _ensure_rgb(frame)
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)


def apply_roi_and_resize(
    frame: np.ndarray,
    roi: str = "full",
    size: int = 72,
    bbox: Optional[BBox] = None,
    features: Optional[dict[str, tuple[float, float] | tuple[float, float, float, float] | list[tuple[float, float]]]] = None,
) -> np.ndarray:
    frame = _ensure_rgb(frame)
    height, width = frame.shape[:2]

    if roi == "full":
        roi_frame = frame
    elif roi == "ellipse":
        if bbox is None:
            roi_frame = frame
        else:
            y_slice, x_slice = _bbox_to_slices(bbox, height, width)
            roi_frame = frame[y_slice, x_slice]
            local_bbox = (0, 0, roi_frame.shape[1], roi_frame.shape[0])
            roi_frame = _apply_ellipse_mask(roi_frame, local_bbox)
    elif roi == "bbox":
        # Keep a compact face crop only; no internal mask is drawn for this mode.
        y_slice, x_slice = _bbox_to_slices(bbox, height, width)
        roi_frame = frame[y_slice, x_slice]
    elif roi in {"haar", "mediapipe"}:
        y_slice, x_slice = _bbox_to_slices(bbox, height, width)
        roi_frame = frame[y_slice, x_slice]
    else:
        raise ValueError(f"Unsupported ROI strategy: {roi}")

    return _resize_frame(roi_frame, size)


def extract_roi(frame: np.ndarray, roi: str = "bbox", size: int = 72, bbox: Optional[BBox] = None) -> np.ndarray:
    return apply_roi_and_resize(frame, roi=roi, size=size, bbox=bbox)


def detect_face_bbox(
    frames: Sequence[np.ndarray],
    roi: str = "bbox",
    pad: float = 0.0,
    forehead_ratio: float = 0.20,
    side_ratio: float = 0.20,
    bottom_ratio: float = 0.20,
    max_frames: int = 30,
) -> Optional[BBox]:
    if roi == "haar":
        return detect_face_bbox_haar(
            frames,
            pad=pad,
            forehead_ratio=forehead_ratio,
            side_ratio=side_ratio,
            bottom_ratio=bottom_ratio,
            max_frames=max_frames,
        )
    if roi == "mediapipe":
        return detect_face_bbox_mediapipe(frames, pad=pad, max_frames=max_frames)
    if roi == "bbox":
        base_bbox = detect_face_bbox_mediapipe(frames, pad=pad, max_frames=max_frames) or detect_face_bbox_haar(
            frames,
            pad=pad,
            forehead_ratio=0.0,
            side_ratio=0.0,
            bottom_ratio=0.0,
            max_frames=max_frames,
        ) or _default_face_bbox(frames[0])
        h, w = frames[0].shape[:2]
        return _expand_bbox_directional(
            base_bbox,
            forehead_ratio=forehead_ratio,
            side_ratio=side_ratio,
            bottom_ratio=bottom_ratio,
            width=w,
            height=h,
        )
    return None


def transform_frames_with_roi(
    frames: np.ndarray,
    roi: str = "bbox",
    size: int = 72,
    pad: float = 0.0,
    forehead_ratio: float = 0.20,
    side_ratio: float = 0.20,
    bottom_ratio: float = 0.20,
) -> tuple[np.ndarray, Optional[BBox]]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [N, H, W, 3], got {frames.shape}")

    bbox = detect_face_bbox(
        frames,
        roi=roi,
        pad=pad,
        forehead_ratio=forehead_ratio,
        side_ratio=side_ratio,
        bottom_ratio=bottom_ratio,
    )
    processed = [
        apply_roi_and_resize(frame, roi=roi, size=size, bbox=bbox)
        for frame in frames
    ]
    return np.stack(processed, axis=0), bbox


def get_face_bbox(*args, **kwargs):
    return detect_face_bbox(*args, **kwargs)


def roi_extract(*args, **kwargs):
    return extract_roi(*args, **kwargs)


def crop_and_resize_roi(*args, **kwargs):
    return apply_roi_and_resize(*args, **kwargs)


__all__ = [
    "BBox",
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