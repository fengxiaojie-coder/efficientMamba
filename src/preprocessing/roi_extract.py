from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]

# Alignment safeguards: avoid over-correcting near-frontal faces.
MAX_ROLL_CORRECTION_DEG = 20.0
ROLL_DEADZONE_DEG = 4.0
MAX_SCALE_CORRECTION = 1.20

# Ordered MediaPipe face-oval landmark indices (clockwise loop).
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


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

    if not hasattr(mp, 'solutions') or getattr(mp, 'solutions', None) is None:
        logger.warning('mediapipe package is present but missing solutions API; falling back to non-mediapipe ROI')
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


def detect_face_polygon_mediapipe(
    frames: Sequence[np.ndarray],
    max_frames: int = 30,
    forehead_ratio: float = 0.20,
    side_ratio: float = 0.0,
    bottom_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    """Detect a stable face contour polygon from MediaPipe face landmarks."""
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning('mediapipe is not installed; face_mesh ROI unavailable')
        return None

    if not hasattr(mp, 'solutions') or getattr(mp, 'solutions', None) is None:
        logger.warning('mediapipe package is present but missing solutions API; face_mesh ROI unavailable')
        return None

    rows: list[np.ndarray] = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        for frame in list(frames)[:max_frames]:
            frame = _ensure_rgb(frame)
            h, w = frame.shape[:2]
            result = face_mesh.process(frame)
            if not result.multi_face_landmarks:
                continue
            lm = result.multi_face_landmarks[0].landmark
            all_pts = np.asarray([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
            pts = []
            for idx in FACE_OVAL_LANDMARKS:
                p = lm[idx]
                x = int(round(p.x * w))
                y = int(round(p.y * h))
                pts.append((x, y))
            arr = np.asarray(pts, dtype=np.float32)
            arr = _relocate_cheek_band_points(arr, all_pts, side_ratio)

            rows.append(arr)
    finally:
        face_mesh.close()

    if not rows:
        return None

    poly = np.median(np.stack(rows, axis=0), axis=0)

    # Extend forehead coverage by lifting top-most contour points slightly.
    y_min = float(np.min(poly[:, 1]))
    y_max = float(np.max(poly[:, 1]))
    x_min = float(np.min(poly[:, 0]))
    x_max = float(np.max(poly[:, 0]))
    x_mid = float(np.mean(poly[:, 0]))
    face_h = max(1.0, y_max - y_min)
    face_w = max(1.0, x_max - x_min)
    lift = float(forehead_ratio) * face_h
    top_mask = poly[:, 1] <= (y_min + 0.22 * face_h)
    poly[top_mask, 1] = poly[top_mask, 1] - lift

    # side_ratio is applied in cheek band during per-frame point extraction.

    # bottom_ratio < 0 trims chin/neck; > 0 keeps more lower face.
    if abs(float(bottom_ratio)) > 1e-8:
        bottom_mask = poly[:, 1] >= (y_min + 0.72 * face_h)
        poly[bottom_mask, 1] = poly[bottom_mask, 1] + float(bottom_ratio) * face_h

    h, w = frames[0].shape[:2]
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    return np.round(poly).astype(np.int32)


def detect_face_polygons_mediapipe(
    frames: Sequence[np.ndarray],
    forehead_ratio: float = 0.20,
    side_ratio: float = 0.0,
    bottom_ratio: float = 0.0,
    smoothing_alpha: float = 0.65,
    static_image_mode: bool = False,
    fallback_to_previous: bool = True,
    return_alignment: bool = False,
    return_scale: bool = False,
) -> list[Optional[np.ndarray]] | tuple[list[Optional[np.ndarray]], list[Optional[tuple[float, float]]], list[Optional[float]]] | tuple[list[Optional[np.ndarray]], list[Optional[tuple[float, float]]], list[Optional[float]], list[Optional[float]]]:
    """Detect face contour polygon for each frame with temporal smoothing."""
    empty_polys: list[Optional[np.ndarray]] = [None for _ in frames]
    empty_centers: list[Optional[tuple[float, float]]] = [None for _ in frames]
    empty_angles: list[Optional[float]] = [None for _ in frames]
    empty_scales: list[Optional[float]] = [None for _ in frames]

    try:
        import mediapipe as mp
    except ImportError:
        logger.warning('mediapipe is not installed; face_mesh ROI unavailable')
        if return_alignment:
            if return_scale:
                return empty_polys, empty_centers, empty_angles, empty_scales
            return empty_polys, empty_centers, empty_angles
        return empty_polys

    if not hasattr(mp, 'solutions') or getattr(mp, 'solutions', None) is None:
        logger.warning('mediapipe package is present but missing solutions API; face_mesh ROI unavailable')
        if return_alignment:
            if return_scale:
                return empty_polys, empty_centers, empty_angles, empty_scales
            return empty_polys, empty_centers, empty_angles
        return empty_polys

    polys: list[Optional[np.ndarray]] = []
    align_centers: list[Optional[tuple[float, float]]] = []
    align_angles: list[Optional[float]] = []
    align_scales: list[Optional[float]] = []
    prev: Optional[np.ndarray] = None
    prev_center: Optional[tuple[float, float]] = None
    prev_angle: Optional[float] = None
    prev_scale: Optional[float] = None
    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=bool(static_image_mode),
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        for frame in frames:
            frame = _ensure_rgb(frame)
            h, w = frame.shape[:2]
            result = face_mesh.process(frame)

            if not result.multi_face_landmarks:
                if fallback_to_previous:
                    polys.append(None if prev is None else np.round(prev).astype(np.int32))
                    align_centers.append(prev_center)
                    align_angles.append(prev_angle)
                    align_scales.append(prev_scale)
                else:
                    polys.append(None)
                    align_centers.append(None)
                    align_angles.append(None)
                    align_scales.append(None)
                continue

            lm = result.multi_face_landmarks[0].landmark
            pts = []
            for idx in FACE_OVAL_LANDMARKS:
                p = lm[idx]
                x = float(p.x * w)
                y = float(p.y * h)
                pts.append((x, y))
            poly = np.asarray(pts, dtype=np.float32)
            all_pts = np.asarray([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

            y_min = float(np.min(poly[:, 1]))
            y_max = float(np.max(poly[:, 1]))
            face_h = max(1.0, y_max - y_min)

            # Forehead lift.
            lift = float(forehead_ratio) * face_h
            top_mask = poly[:, 1] <= (y_min + 0.22 * face_h)
            poly[top_mask, 1] = poly[top_mask, 1] - lift

            # Only relocate contour points in cheek height band.
            poly = _relocate_cheek_band_points(poly, all_pts, side_ratio)

            # bottom_ratio < 0 trims chin/neck; > 0 keeps more lower face.
            if abs(float(bottom_ratio)) > 1e-8:
                bottom_mask = poly[:, 1] >= (y_min + 0.72 * face_h)
                poly[bottom_mask, 1] = poly[bottom_mask, 1] + float(bottom_ratio) * face_h

            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

            # Roll alignment info: use eye-line as primary signal, nose-axis as fallback.
            p_top = lm[168]
            p_bottom = lm[6]
            nose_cx = float(lm[1].x * w)
            nose_cy = float(lm[1].y * h)
            dx = float((p_bottom.x - p_top.x) * w)
            dy = float((p_bottom.y - p_top.y) * h)
            if abs(dx) + abs(dy) > 1e-6:
                axis_angle = float(np.degrees(np.arctan2(dy, dx)))
                nose_rot = 90.0 - axis_angle
            else:
                nose_rot = None

            # Use eye corners to estimate roll; this is more stable than eyelid means.
            # Left eye corners: outer=33, inner=133; right eye corners: inner=362, outer=263.
            lx = float(((lm[33].x + lm[133].x) * 0.5) * w)
            ly = float(((lm[33].y + lm[133].y) * 0.5) * h)
            rx = float(((lm[362].x + lm[263].x) * 0.5) * w)
            ry = float(((lm[362].y + lm[263].y) * 0.5) * h)
            ex = rx - lx
            ey = ry - ly
            if abs(ex) + abs(ey) > 1e-6:
                eye_angle = float(np.degrees(np.arctan2(ey, ex)))
                eye_rot = eye_angle
                eye_dist = float(np.sqrt(ex * ex + ey * ey))
            else:
                eye_rot = None
                eye_dist = None

            # Prefer eye-line midpoint as translation center; fallback to nose tip.
            if eye_rot is not None:
                center_x = 0.5 * (lx + rx)
                center_y = 0.5 * (ly + ry)
            else:
                center_x = nose_cx
                center_y = nose_cy

            if eye_rot is not None and nose_rot is not None:
                # Eye-line gives robust roll; keep some nose-axis influence.
                rot_angle = 0.75 * eye_rot + 0.25 * nose_rot
            elif eye_rot is not None:
                rot_angle = eye_rot
            elif nose_rot is not None:
                rot_angle = nose_rot
            else:
                rot_angle = None

            if rot_angle is not None:
                rot_angle = float(np.clip(rot_angle, -MAX_ROLL_CORRECTION_DEG, MAX_ROLL_CORRECTION_DEG))
                if abs(rot_angle) < ROLL_DEADZONE_DEG:
                    rot_angle = 0.0
                center = (center_x, center_y)
            else:
                if fallback_to_previous:
                    center = prev_center
                    rot_angle = prev_angle
                else:
                    center = None
                    rot_angle = None

            if prev is not None:
                poly = alpha * poly + (1.0 - alpha) * prev
            if prev_center is not None and center is not None:
                center = (
                    alpha * center[0] + (1.0 - alpha) * prev_center[0],
                    alpha * center[1] + (1.0 - alpha) * prev_center[1],
                )
            if prev_angle is not None and rot_angle is not None:
                rot_angle = alpha * rot_angle + (1.0 - alpha) * prev_angle

            scale_val = eye_dist
            if fallback_to_previous and prev_scale is not None and scale_val is not None:
                scale_val = alpha * scale_val + (1.0 - alpha) * prev_scale
            elif fallback_to_previous and prev_scale is not None and scale_val is None:
                scale_val = prev_scale

            prev = poly
            prev_center = center
            prev_angle = rot_angle
            prev_scale = scale_val
            polys.append(np.round(poly).astype(np.int32))
            align_centers.append(center)
            align_angles.append(rot_angle)
            align_scales.append(scale_val)
    finally:
        face_mesh.close()

    if return_alignment:
        if return_scale:
            return polys, align_centers, align_angles, align_scales
        return polys, align_centers, align_angles
    return polys


def _rotate_frame_and_polygon(
    frame: np.ndarray,
    poly: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    poly_f = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    rot_poly = cv2.transform(poly_f, M).reshape(-1, 2)
    rot_poly[:, 0] = np.clip(rot_poly[:, 0], 0, w - 1)
    rot_poly[:, 1] = np.clip(rot_poly[:, 1], 0, h - 1)
    return rotated, np.round(rot_poly).astype(np.int32)


def _translate_frame_and_polygon(
    frame: np.ndarray,
    poly: np.ndarray,
    center: tuple[float, float],
    target_center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    dx = float(target_center[0] - center[0])
    dy = float(target_center[1] - center[1])
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    poly_f = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    shifted_poly = cv2.transform(poly_f, M).reshape(-1, 2)
    shifted_poly[:, 0] = np.clip(shifted_poly[:, 0], 0, w - 1)
    shifted_poly[:, 1] = np.clip(shifted_poly[:, 1], 0, h - 1)
    return shifted, np.round(shifted_poly).astype(np.int32)


def _scale_frame_and_polygon(
    frame: np.ndarray,
    poly: np.ndarray,
    center: tuple[float, float],
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    s = float(scale_factor)
    M = cv2.getRotationMatrix2D(center, 0.0, s)
    scaled = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    poly_f = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    scaled_poly = cv2.transform(poly_f, M).reshape(-1, 2)
    scaled_poly[:, 0] = np.clip(scaled_poly[:, 0], 0, w - 1)
    scaled_poly[:, 1] = np.clip(scaled_poly[:, 1], 0, h - 1)
    return scaled, np.round(scaled_poly).astype(np.int32)


def _poly_to_bbox(poly: np.ndarray, width: int, height: int) -> BBox:
    x1 = int(np.clip(np.min(poly[:, 0]), 0, width - 1))
    x2 = int(np.clip(np.max(poly[:, 0]) + 1, 1, width))
    y1 = int(np.clip(np.min(poly[:, 1]), 0, height - 1))
    y2 = int(np.clip(np.max(poly[:, 1]) + 1, 1, height))
    return _clip_bbox(x1, y1, x2 - x1, y2 - y1, width, height)


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
    elif roi == "face_mesh":
        poly = None
        if isinstance(features, dict):
            poly = features.get('face_polygon')
        if poly is None:
            # Fallback to bbox crop when landmarks are unavailable.
            y_slice, x_slice = _bbox_to_slices(bbox, height, width)
            roi_frame = frame[y_slice, x_slice]
        else:
            poly = np.asarray(poly, dtype=np.int32)
            x1 = int(np.clip(np.min(poly[:, 0]), 0, width - 1))
            x2 = int(np.clip(np.max(poly[:, 0]) + 1, 1, width))
            y1 = int(np.clip(np.min(poly[:, 1]), 0, height - 1))
            y2 = int(np.clip(np.max(poly[:, 1]) + 1, 1, height))
            if x2 <= x1 or y2 <= y1:
                y_slice, x_slice = _bbox_to_slices(bbox, height, width)
                roi_frame = frame[y_slice, x_slice]
            else:
                crop = frame[y1:y2, x1:x2]
                local = poly.copy()
                local[:, 0] -= x1
                local[:, 1] -= y1
                mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [local], 255)
                roi_frame = cv2.bitwise_and(crop, crop, mask=mask)
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
    if roi in {"bbox", "face_mesh"}:
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
    if roi == "ellipse":
        # Ellipse masking still needs a face bbox; reuse bbox detection pipeline.
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
    align_nose_axis: bool = False,
    face_mesh_first_frame_mask: bool = False,
    center_face: bool = False,
    canonical_face_mask: bool = False,
    frame_independent_face_mesh: bool = False,
) -> tuple[np.ndarray, Optional[BBox]]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [N, H, W, 3], got {frames.shape}")

    if canonical_face_mask:
        align_nose_axis = True
        center_face = True

    bbox = detect_face_bbox(
        frames,
        roi=roi,
        pad=pad,
        forehead_ratio=forehead_ratio,
        side_ratio=side_ratio,
        bottom_ratio=bottom_ratio,
    )
    processed: list[np.ndarray] = []
    if roi == 'face_mesh':
        h, w = frames[0].shape[:2]
        canvas_center = (w / 2.0, h / 2.0)
        if face_mesh_first_frame_mask:
            static_poly = detect_face_polygon_mediapipe(
                [frames[0]],
                max_frames=1,
                forehead_ratio=forehead_ratio,
                side_ratio=side_ratio,
                bottom_ratio=bottom_ratio,
            )
            static_center: Optional[tuple[float, float]] = None
            static_angle: Optional[float] = None
            static_scale: Optional[float] = None
            frame_centers: list[Optional[tuple[float, float]]] = [None for _ in frames]
            frame_angles: list[Optional[float]] = [None for _ in frames]
            frame_scales: list[Optional[float]] = [None for _ in frames]
            if align_nose_axis:
                # frame_independent_face_mesh uses static_image_mode=True which already
                # treats every frame independently (no tracking state).  A single FaceMesh
                # instance is reused to avoid per-frame GPU context initialisation.
                _polys_all, centers_all, angles_all, scales_all = detect_face_polygons_mediapipe(
                    frames,
                    forehead_ratio=forehead_ratio,
                    side_ratio=side_ratio,
                    bottom_ratio=bottom_ratio,
                    smoothing_alpha=1.0,
                    static_image_mode=frame_independent_face_mesh,
                    fallback_to_previous=not frame_independent_face_mesh,
                    return_alignment=True,
                    return_scale=True,
                )
                frame_centers = centers_all
                frame_angles = angles_all
                frame_scales = scales_all
                if centers_all:
                    static_center = centers_all[0]
                if angles_all:
                    static_angle = angles_all[0]
                if scales_all:
                    static_scale = scales_all[0]

            if static_poly is not None:
                # Build fixed mask in canonical coordinates using first-frame registration.
                canonical_poly = static_poly
                _dummy = frames[0]
                if align_nose_axis and static_center is not None and static_angle is not None and abs(float(static_angle)) > 1e-4:
                    _dummy, canonical_poly = _rotate_frame_and_polygon(_dummy, canonical_poly, static_center, static_angle)
                if center_face:
                    base_center = static_center if static_center is not None else (
                        float(np.mean(canonical_poly[:, 0])), float(np.mean(canonical_poly[:, 1]))
                    )
                    _dummy, canonical_poly = _translate_frame_and_polygon(_dummy, canonical_poly, base_center, canvas_center)

                bbox = _poly_to_bbox(canonical_poly, w, h)
                last_valid_center = static_center
                last_valid_angle = static_angle
                last_valid_scale = static_scale
                for i, frame in enumerate(frames):
                    curr_frame = frame
                    curr_poly = canonical_poly
                    curr_center = frame_centers[i] if i < len(frame_centers) else None
                    curr_angle = frame_angles[i] if i < len(frame_angles) else None
                    curr_scale = frame_scales[i] if i < len(frame_scales) else None

                    # If a frame misses landmarks, keep geometric registration stable by
                    # reusing the latest valid alignment parameters.
                    if curr_center is None:
                        curr_center = last_valid_center
                    else:
                        last_valid_center = curr_center
                    if curr_angle is None:
                        curr_angle = last_valid_angle
                    else:
                        last_valid_angle = curr_angle
                    if curr_scale is None:
                        curr_scale = last_valid_scale
                    else:
                        last_valid_scale = curr_scale

                    if align_nose_axis and curr_center is not None and curr_angle is not None and abs(float(curr_angle)) > 1e-4:
                        curr_frame, _tmp_poly = _rotate_frame_and_polygon(curr_frame, static_poly, curr_center, curr_angle)

                    # Optional scale normalization to first-frame eye distance.
                    if align_nose_axis and static_scale is not None and curr_scale is not None and curr_center is not None and curr_scale > 1e-6:
                        sf = float(np.clip(static_scale / curr_scale, 1.0 / MAX_SCALE_CORRECTION, MAX_SCALE_CORRECTION))
                        if abs(sf - 1.0) > 1e-4:
                            curr_frame, _tmp_poly = _scale_frame_and_polygon(curr_frame, static_poly, curr_center, sf)

                    if center_face:
                        if curr_center is not None:
                            curr_frame, _tmp_poly = _translate_frame_and_polygon(curr_frame, static_poly, curr_center, canvas_center)
                    processed.append(
                        apply_roi_and_resize(
                            curr_frame,
                            roi=roi,
                            size=size,
                            bbox=_poly_to_bbox(curr_poly, w, h),
                            features={'face_polygon': curr_poly},
                        )
                    )
                return np.stack(processed, axis=0), bbox

        if align_nose_axis:
            frame_polys, align_centers, align_angles = detect_face_polygons_mediapipe(
                frames,
                forehead_ratio=forehead_ratio,
                side_ratio=side_ratio,
                bottom_ratio=bottom_ratio,
                static_image_mode=frame_independent_face_mesh,
                fallback_to_previous=not frame_independent_face_mesh,
                return_alignment=True,
            )
        else:
            frame_polys = detect_face_polygons_mediapipe(
                frames,
                forehead_ratio=forehead_ratio,
                side_ratio=side_ratio,
                bottom_ratio=bottom_ratio,
                static_image_mode=frame_independent_face_mesh,
                fallback_to_previous=not frame_independent_face_mesh,
            )
            align_centers = [None for _ in frames]
            align_angles = [None for _ in frames]
        frame_bboxes: list[BBox] = []
        last_poly: Optional[np.ndarray] = None
        last_center: Optional[tuple[float, float]] = None
        last_angle: Optional[float] = None
        for i, frame in enumerate(frames):
            poly = frame_polys[i] if i < len(frame_polys) else None
            center = align_centers[i] if i < len(align_centers) else None
            angle = align_angles[i] if i < len(align_angles) else None
            if poly is None:
                poly = last_poly
                center = last_center
                angle = last_angle
            if poly is not None:
                last_poly = poly
                if center is not None:
                    last_center = center
                if angle is not None:
                    last_angle = angle
                curr_frame = frame
                if align_nose_axis and center is not None and angle is not None and abs(float(angle)) > 1e-4:
                    curr_frame, poly = _rotate_frame_and_polygon(frame, poly, center, angle)
                if center_face:
                    align_center = center if center is not None else (
                        float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
                    )
                    curr_frame, poly = _translate_frame_and_polygon(curr_frame, poly, align_center, canvas_center)
                curr_bbox = _poly_to_bbox(poly, w, h)
                frame_bboxes.append(curr_bbox)
                curr_features = {'face_polygon': poly}
            else:
                curr_frame = frame
                curr_bbox = bbox
                curr_features = None
            processed.append(apply_roi_and_resize(curr_frame, roi=roi, size=size, bbox=curr_bbox, features=curr_features))
        if frame_bboxes:
            bbox = _median_bbox(frame_bboxes)
    else:
        processed = [
            apply_roi_and_resize(frame, roi=roi, size=size, bbox=bbox, features=None)
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


def _relocate_cheek_band_points(poly: np.ndarray, all_pts: np.ndarray, side_ratio: float) -> np.ndarray:
    """Apply side_ratio only on lower-half contour points (jaw/lower cheek)."""
    if abs(float(side_ratio)) <= 1e-8:
        return poly

    _ = all_pts  # kept for backward-compatible signature
    refined = poly.copy()
    y_min = float(np.min(refined[:, 1]))
    y_max = float(np.max(refined[:, 1]))
    x_mid = float(np.mean(refined[:, 0]))
    face_h = max(1.0, y_max - y_min)

    # Only adjust lower half circle to avoid masking upper cheek area.
    y_lo = y_min + 0.56 * face_h
    y_hi = y_min + 0.98 * face_h
    base = float(np.clip(abs(float(side_ratio)), 0.0, 0.35))

    for i in range(refined.shape[0]):
        x = float(refined[i, 0])
        y = float(refined[i, 1])
        if y < y_lo or y > y_hi:
            continue

        # Stronger effect near chin, weaker near mid-face.
        t = float(np.clip((y - y_lo) / max(1e-6, (y_hi - y_lo)), 0.0, 1.0))
        local = base * (0.35 + 0.65 * t)
        if float(side_ratio) < 0:
            scale = max(0.45, 1.0 - local)
        else:
            scale = min(1.35, 1.0 + local)
        refined[i, 0] = x_mid + (x - x_mid) * scale

    return refined