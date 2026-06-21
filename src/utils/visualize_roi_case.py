"""Create a visual example of ROI selection for UBFC videos.

Example:
    python -m src.utils.visualize_roi_case --sample_dir dataSet/UBFC-rPPG/subject1 --roi bbox

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

from src.preprocessing.roi_extract import transform_frames_with_roi
from src.preprocessing.ubfc_to_tensors import extract_frames


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize an ROI selection example')
    parser.add_argument('--sample_dir', required=True, help='UBFC subject directory, e.g. dataSet/UBFC-rPPG/subject1')
    parser.add_argument('--roi', type=str, default='bbox', choices=['full', 'haar', 'mediapipe', 'ellipse', 'bbox'])
    parser.add_argument('--roi_pad', type=float, default=0.0)
    parser.add_argument('--forehead_ratio', type=float, default=0.20,
                        help='Extra top expansion ratio for bbox/haar ROI (relative to face-box height)')
    parser.add_argument('--size', type=int, default=72)
    parser.add_argument('--out', type=str, default='results/roi_visualization')
    parser.add_argument('--frame_index', type=int, default=0, help='Which frame to visualize')
    args = parser.parse_args()

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
    frames, _fps = extract_frames(video_path, img_size=args.size)

    frame_index = max(0, min(int(args.frame_index), frames.shape[0] - 1))
    resized, bbox = transform_frames_with_roi(
        frames,
        roi=args.roi,
        size=args.size,
        pad=args.roi_pad,
        forehead_ratio=args.forehead_ratio,
    )

    raw_frame = frames[frame_index]
    roi_frame = resized[frame_index]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{sample_dir.name}_{args.roi}_roi_example.png'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_frame)
    axes[0].set_title('Original frame')
    axes[0].axis('off')

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

    print(out_path)


if __name__ == '__main__':
    main()
