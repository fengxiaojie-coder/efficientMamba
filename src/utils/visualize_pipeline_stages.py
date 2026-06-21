"""Visualize full rPPG pipeline stages on one UBFC sample.

Stages:
1) Original frame + detected ROI box
2) ROI-resized frame used by preprocessing
3) Motion diff magnitude (middle timestep)
4) G1 attention map (if checkpoint provided)
5) G2 attention map (if checkpoint provided)

Example:
    python -m src.utils.visualize_pipeline_stages \
      --sample_dir dataSet/UBFC-rPPG/subject9 \
      --out results/pipeline_stages \
      --frame_index 20 \
      --clip_len 64 \
      --roi bbox \
      --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from src.models.efficientphys_mamba import EfficientPhysMambaRegressor
from src.preprocessing.roi_extract import transform_frames_with_roi
from src.preprocessing.ubfc_to_tensors import extract_frames, make_clips


def _find_video(sample_dir: Path) -> Path | None:
    for ext in ('.avi', '.mp4', '.mov'):
        candidate = sample_dir / f'vid{ext}'
        if candidate.exists():
            return candidate
    vids = list(sample_dir.glob('*.avi')) + list(sample_dir.glob('*.mp4')) + list(sample_dir.glob('*.mov'))
    return vids[0] if vids else None


def _select_clip_start(total_frames: int, clip_len: int, preferred_frame: int) -> int:
    if total_frames <= clip_len:
        return 0
    half = clip_len // 2
    start = max(0, min(preferred_frame - half, total_frames - clip_len))
    return int(start)


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize full preprocessing->attention stages for one UBFC sample')
    parser.add_argument('--sample_dir', required=True, help='UBFC subject directory, e.g. dataSet/UBFC-rPPG/subject9')
    parser.add_argument('--out', default='results/pipeline_stages', help='Output directory')
    parser.add_argument('--size', type=int, default=72)
    parser.add_argument('--roi', type=str, default='bbox', choices=['full', 'haar', 'mediapipe', 'ellipse', 'bbox'])
    parser.add_argument('--roi_pad', type=float, default=0.0)
    parser.add_argument('--forehead_ratio', type=float, default=0.20)
    parser.add_argument('--frame_index', type=int, default=20)
    parser.add_argument('--clip_len', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='', help='Optional model checkpoint to render G1/G2 attention')
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    video_path = _find_video(sample_dir)
    if video_path is None:
        raise SystemExit(f'No video found under {sample_dir}')

    frames, _fps = extract_frames(video_path, img_size=args.size)
    if frames.shape[0] == 0:
        raise SystemExit(f'No frames extracted from {video_path}')

    frame_index = max(0, min(int(args.frame_index), frames.shape[0] - 1))
    roi_frames, bbox = transform_frames_with_roi(
        frames,
        roi=args.roi,
        size=args.size,
        pad=args.roi_pad,
        forehead_ratio=args.forehead_ratio,
    )

    # Build one clip centered at frame_index for diff/attention visualization.
    start = _select_clip_start(frames.shape[0], int(args.clip_len), frame_index)
    window = roi_frames[start:start + int(args.clip_len)]
    if window.shape[0] < int(args.clip_len):
        pad_n = int(args.clip_len) - window.shape[0]
        if window.shape[0] == 0:
            raise SystemExit('ROI window is empty after extraction')
        tail = np.repeat(window[-1:], pad_n, axis=0)
        window = np.concatenate([window, tail], axis=0)

    clips = make_clips(window, clip_len=int(args.clip_len), stride=int(args.clip_len))
    if clips.shape[0] == 0:
        raise SystemExit('Could not build a clip for visualization')
    clip_t = torch.from_numpy(clips[0]).float().unsqueeze(0)  # [1,T,6,H,W]
    t_mid = clip_t.shape[1] // 2

    # Stage 3: diff magnitude from clip middle timestep
    diff = clip_t[0, t_mid, :3].permute(1, 2, 0).numpy()
    diff_mag = np.linalg.norm(diff, axis=-1)

    g1 = None
    g2_up = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f'Checkpoint not found: {ckpt_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EfficientPhysMambaRegressor(in_channels=6, frame_depth=int(args.clip_len)).to(device)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        with torch.no_grad():
            _, attn = model(clip_t.to(device), return_attention=True)
        g1 = attn['g1'][0, t_mid, 0].detach().cpu().numpy()
        g2 = attn['g2'][0, t_mid, 0].detach().cpu().unsqueeze(0).unsqueeze(0)
        g2_up = F.interpolate(g2, size=g1.shape, mode='bilinear', align_corners=False).squeeze().numpy()

    raw_frame = frames[frame_index]
    roi_frame = roi_frames[frame_index]

    cols = 5 if g1 is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4.2))

    axes[0].imshow(raw_frame)
    axes[0].set_title('Stage 1: original + ROI')
    axes[0].axis('off')
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)

    axes[1].imshow(roi_frame)
    axes[1].set_title('Stage 2: ROI resized')
    axes[1].axis('off')

    im2 = axes[2].imshow(diff_mag, cmap='magma')
    axes[2].set_title('Stage 3: diff magnitude')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if g1 is not None:
        im3 = axes[3].imshow(g1, cmap='hot')
        axes[3].set_title('Stage 4: G1 attention')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(g2_up, cmap='hot')
        axes[4].set_title('Stage 5: G2 attention')
        axes[4].axis('off')
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

    title = f'Pipeline stages | {sample_dir.name} | frame={frame_index} | roi={args.roi} | forehead_ratio={args.forehead_ratio}'
    if args.checkpoint:
        title += f' | ckpt={Path(args.checkpoint).name}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{sample_dir.name}_pipeline_stage.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(out_path)


if __name__ == '__main__':
    main()
