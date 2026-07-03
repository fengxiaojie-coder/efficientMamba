"""Create a visual example of ROI selection for UBFC videos.

Example:
    python -m src.utils.visualize_roi_case --sample_dir dataSet/UBFC-rPPG/subject1 --roi bbox

To visualize what was actually saved during preprocessing, use --clip_pt:
    python -m src.utils.visualize_roi_case --clip_pt /path/to/subject_clip_0000.pt --frame_index 64

If MediaPipe is not installed, use --roi haar, --roi ellipse, --roi bbox, or --roi full.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from src.preprocessing.roi_extract import transform_frames_with_roi
from src.preprocessing.ubfc_to_tensors import extract_frames


def _detect_eye_corner_line(frame: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    try:
        import mediapipe as mp
    except ImportError:
        return None

    if not hasattr(mp, 'solutions') or getattr(mp, 'solutions', None) is None:
        return None

    h, w = frame.shape[:2]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    try:
        result = face_mesh.process(frame)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        left = (
            float(((lm[33].x + lm[133].x) * 0.5) * w),
            float(((lm[33].y + lm[133].y) * 0.5) * h),
        )
        right = (
            float(((lm[362].x + lm[263].x) * 0.5) * w),
            float(((lm[362].y + lm[263].y) * 0.5) * h),
        )
        angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
        return left, right, angle
    finally:
        face_mesh.close()


def _load_clip_pt_frame(clip_pt: Path, frame_index: int) -> np.ndarray:
    """Load a preprocessed .pt clip and return the raw RGB frame at frame_index.

    The clip tensor has shape [T, 6, H, W].  Channels 3:6 are the normalised
    raw RGB values (float32, 0-1).  We convert them back to uint8 [H,W,3].
    """
    d = torch.load(str(clip_pt), weights_only=False)
    clip = d['clip']  # [T, 6, H, W]
    T = clip.shape[0]
    idx = max(0, min(int(frame_index), T - 1))
    raw = clip[idx, 3:6].permute(1, 2, 0).numpy()  # [H, W, 3]
    raw = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize an ROI selection example')
    parser.add_argument('--sample_dir', default=None, help='UBFC subject directory, e.g. dataSet/UBFC-rPPG/subject1')
    parser.add_argument('--clip_pt', default=None,
                        help='Path to a preprocessed .pt clip file. When given, visualize the actual stored frames '
                             'instead of re-running the ROI pipeline from the raw video.')
    parser.add_argument('--roi', type=str, default='bbox', choices=['full', 'haar', 'mediapipe', 'ellipse', 'bbox', 'face_mesh'])
    parser.add_argument('--decoder', type=str, default='auto', choices=['auto', 'torchvision', 'opencv'],
                        help='Video decoder backend. Use opencv when torchvision is unavailable.')
    parser.add_argument('--roi_pad', type=float, default=0.0)
    parser.add_argument('--forehead_ratio', type=float, default=0.20,
                        help='Extra top expansion ratio for bbox/haar ROI (relative to face-box height)')
    parser.add_argument('--side_ratio', type=float, default=0.0,
                        help='Horizontal expansion ratio for ROI. For face_mesh, negative values shrink lower-half contour.')
    parser.add_argument('--bottom_ratio', type=float, default=0.0,
                        help='Bottom expansion ratio for ROI. For face_mesh, negative values trim chin/neck.')
    parser.add_argument('--align_nose_axis', action='store_true',
                        help='For face_mesh ROI: rotate each frame so nose bridge axis is vertical.')
    parser.add_argument('--face_mesh_first_frame_mask', action='store_true',
                        help='For face_mesh ROI: detect mask from first frame only and reuse for all frames.')
    parser.add_argument('--center_face', action='store_true',
                        help='For face_mesh ROI: translate the nose center to the image center after alignment.')
    parser.add_argument('--canonical_face_mask', action='store_true',
                        help='For face_mesh ROI: enable align_nose_axis + center_face + face_mesh_first_frame_mask together.')
    parser.add_argument('--frame_independent_face_mesh', action='store_true',
                        help='For face_mesh ROI: run per-frame independent detection/alignment (no cross-frame shared detector state).')
    parser.add_argument('--size', type=int, default=72)
    parser.add_argument('--out', type=str, default='results/roi_visualization')
    parser.add_argument('--frame_index', type=int, default=0, help='Which frame to visualize')
    parser.add_argument(
        '--single_frame_only',
        action='store_true',
        help='Run ROI transform on only the selected frame (for context-independence debugging).',
    )
    args = parser.parse_args()

    if args.clip_pt is None and args.sample_dir is None:
        raise SystemExit('Provide either --sample_dir (raw video) or --clip_pt (preprocessed .pt clip).')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode A: read from a preprocessed .pt clip ───────────────────────────
    if args.clip_pt is not None:
        clip_pt = Path(args.clip_pt)
        if not clip_pt.exists():
            raise SystemExit(f'clip_pt not found: {clip_pt}')

        d = torch.load(str(clip_pt), weights_only=False)
        clip = d['clip']  # [T, 6, H, W]
        T = clip.shape[0]
        frame_index = max(0, min(int(args.frame_index), T - 1))

        roi_frame = _load_clip_pt_frame(clip_pt, frame_index)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(roi_frame)
        axes[0].set_title(f'Stored ROI frame {frame_index} (from .pt clip)')
        axes[0].axis('off')

        # show diff magnitude of same frame
        diff = clip[frame_index, :3].permute(1, 2, 0).numpy()
        diff_mag = np.linalg.norm(diff, axis=-1)
        im = axes[1].imshow(diff_mag, cmap='magma')
        axes[1].set_title('Diff magnitude (channels 0:3)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        subject_name = clip_pt.stem
        fig.suptitle(f'Preprocessed clip: {subject_name} | frame={frame_index}', fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f'{subject_name}_frame{frame_index:04d}_from_pt.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(out_path)
        return

    # ── Mode B: re-run ROI pipeline from raw video (original behaviour) ─────
    sample_dir = Path(args.sample_dir)
    video_path = None
    for ext in ('.avi', '.mp4', '.mov'):
        candidate = sample_dir / f'vid{ext}'
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        vids = list(sample_dir.glob('*.avi')) + list(sample_dir.glob('*.mp4')) + list(sample_dir.glob('*.mov'))
        video_path = vids[0] if vids else None
    if video_path is None:
        raise SystemExit(f'No video found under {sample_dir}')
    frames, _fps = extract_frames(video_path, img_size=args.size, decoder=args.decoder)

    frame_index = max(0, min(int(args.frame_index), frames.shape[0] - 1))
    frames_for_transform = frames if not args.single_frame_only else frames[frame_index:frame_index + 1]
    resized, bbox = transform_frames_with_roi(
        frames_for_transform,
        roi=args.roi,
        size=args.size,
        pad=args.roi_pad,
        forehead_ratio=args.forehead_ratio,
        side_ratio=args.side_ratio,
        bottom_ratio=args.bottom_ratio,
        align_nose_axis=args.align_nose_axis,
        face_mesh_first_frame_mask=args.face_mesh_first_frame_mask,
        center_face=args.center_face,
        canonical_face_mask=args.canonical_face_mask,
        frame_independent_face_mesh=args.frame_independent_face_mesh,
    )

    raw_frame = frames[frame_index]
    roi_frame = resized[0] if args.single_frame_only else resized[frame_index]
    eye_line = _detect_eye_corner_line(raw_frame) if args.roi == 'face_mesh' else None

    out_path = out_dir / f'{sample_dir.name}_{args.roi}_roi_example.png'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_frame)
    axes[0].set_title('Original frame')
    axes[0].axis('off')

    if eye_line is not None:
        left_eye, right_eye, eye_angle = eye_line
        axes[0].plot([left_eye[0], right_eye[0]], [left_eye[1], right_eye[1]], color='cyan', linewidth=2)
        axes[0].scatter([left_eye[0], right_eye[0]], [left_eye[1], right_eye[1]], color='cyan', s=18)
        axes[0].text(8, 36, f'eye-line angle={eye_angle:.2f} deg', color='cyan', fontsize=10, weight='bold')

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, max(0, y1 - 6), f'ROI: {args.roi}', color='lime', fontsize=10, weight='bold')
    else:
        axes[0].text(8, 18, f'ROI: {args.roi} (fallback/full)', color='yellow', fontsize=10, weight='bold')

    axes[1].imshow(roi_frame)
    axes[1].set_title(f'Cropped/resized ROI ({args.size}x{args.size})')
    axes[1].axis('off')

    axes[2].imshow(np.clip(raw_frame.astype(np.float32) * 0.7 + 40, 0, 255).astype(np.uint8))
    axes[2].set_title('Visualization note')
    axes[2].axis('off')
    note = (
        'Green box = detected face ROI\n'
        'Ellipse ROI blacks out outside-face regions\n'
        'Face_mesh ROI uses landmark contour points\n'
        'BBox/haar ROI keep a compact face crop\n'
        'Forehead_ratio extends the top edge upward\n'
        'Right = resized ROI used by model'
    )
    axes[2].text(
        0.02,
        0.05,
        note,
        transform=axes[2].transAxes,
        fontsize=10,
        color='white',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.4'),
    )

    fig.suptitle(f'ROI visual case: {sample_dir.name} | roi={args.roi} | frame={frame_index}', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    if eye_line is not None:
        left_eye, right_eye, eye_angle = eye_line
        print(f'eye_line_left=({left_eye[0]:.2f}, {left_eye[1]:.2f})')
        print(f'eye_line_right=({right_eye[0]:.2f}, {right_eye[1]:.2f})')
        print(f'eye_line_angle_deg={eye_angle:.4f}')

    print(out_path)


if __name__ == '__main__':
    main()
