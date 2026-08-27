from __future__ import annotations

from typing import Optional

import numpy as np

from src.preprocessing.roi.roi_extract import (
    BBox,
    MAX_SCALE_CORRECTION,
    crop_polygon_region,
    detect_compact_face_bbox,
    detect_face_polygon_mediapipe,
    detect_face_polygons_mediapipe,
    _poly_to_bbox,
    _rotate_frame_and_polygon,
    _scale_frame_and_polygon,
    _translate_frame_and_polygon,
)
from .base import BaseROITransformStrategy


class FaceMeshROITransformStrategy(BaseROITransformStrategy):
    name = 'face_mesh'

    def detect_bbox(self, frames, pad=0.0, forehead_ratio=0.20, side_ratio=0.20, bottom_ratio=0.20, max_frames=30):
        return detect_compact_face_bbox(frames, pad=pad, forehead_ratio=forehead_ratio, side_ratio=side_ratio, bottom_ratio=bottom_ratio, max_frames=max_frames)

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
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames with shape [N, H, W, 3], got {frames.shape}")
        if canonical_face_mask:
            align_nose_axis = True
            center_face = True

        bbox = self.detect_bbox(
            frames,
            pad=pad,
            forehead_ratio=forehead_ratio,
            side_ratio=side_ratio,
            bottom_ratio=bottom_ratio,
        )
        processed: list[np.ndarray] = []
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

                    if align_nose_axis and static_scale is not None and curr_scale is not None and curr_center is not None and curr_scale > 1e-6:
                        sf = float(np.clip(static_scale / curr_scale, 1.0 / MAX_SCALE_CORRECTION, MAX_SCALE_CORRECTION))
                        if abs(sf - 1.0) > 1e-4:
                            curr_frame, _tmp_poly = _scale_frame_and_polygon(curr_frame, static_poly, curr_center, sf)

                    if center_face:
                        if curr_center is not None:
                            curr_frame, _tmp_poly = _translate_frame_and_polygon(curr_frame, static_poly, curr_center, canvas_center)
                    processed.append(crop_polygon_region(curr_frame, polygon=curr_poly, bbox=_poly_to_bbox(curr_poly, w, h), size=size))
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
                frame_bboxes.append(_poly_to_bbox(poly, w, h))
            else:
                frame_bboxes.append(bbox)
            last_poly = poly
            last_center = center
            last_angle = angle
        for i, frame in enumerate(frames):
            poly = frame_polys[i] if i < len(frame_polys) else None
            center = align_centers[i] if i < len(align_centers) else None
            angle = align_angles[i] if i < len(align_angles) else None
            if poly is None:
                poly = last_poly
                center = last_center
                angle = last_angle
            frame_bbox = frame_bboxes[i]
            if poly is not None and center is not None and angle is not None and abs(float(angle)) > 1e-4:
                frame, poly = _rotate_frame_and_polygon(frame, poly, center, angle)
            if poly is not None and center is not None and center_face:
                frame, poly = _translate_frame_and_polygon(frame, poly, center, canvas_center)
            processed.append(crop_polygon_region(frame, polygon=poly, bbox=frame_bbox, size=size))
        return np.stack(processed, axis=0), bbox

    def apply(self, frame: np.ndarray, bbox: Optional[BBox], size: int = 72, features=None) -> np.ndarray:
        polygon = features.get('face_polygon') if isinstance(features, dict) else None
        return crop_polygon_region(frame, polygon=polygon, bbox=bbox, size=size)
