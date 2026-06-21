#!/usr/bin/env python
"""Demo script to visualize attention heatmaps on dummy data."""

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from src.models.efficientphys_mamba import EfficientPhysMambaRegressor

def demo_attention_visualization():
    """Generate attention visualizations for debugging."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy data (batch of 4 clips)
    batch_size = 4
    num_frames = 64

    # Create model with matching temporal depth.
    model = EfficientPhysMambaRegressor(in_channels=6, frame_depth=num_frames)
    model.to(device)
    model.eval()

    # Create output directory
    demo_dir = Path('checkpoints/demo_attention')
    demo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {demo_dir}")

    # Create dummy data (batch of 4 clips)
    dummy_clips = torch.randn(batch_size, num_frames, 6, 72, 72, device=device)

    print(f"\nGenerating attention visualizations for {batch_size} samples...")

    with torch.no_grad():
        # Forward pass with attention
        hr_pred, attn_dict = model(dummy_clips, return_attention=True)

        g1 = attn_dict['g1']  # [B, T, 1, H1, W1]
        g2 = attn_dict['g2']  # [B, T, 1, H2, W2]

        print(f"  HR predictions: {hr_pred.squeeze().cpu().numpy().round(1)} bpm")
        print(f"  G1 shape: {g1.shape}, range: [{g1.min():.3f}, {g1.max():.3f}]")
        print(f"  G2 shape: {g2.shape}, range: [{g2.min():.3f}, {g2.max():.3f}]")

        # Create visualization
        fig, axes = plt.subplots(batch_size, 4, figsize=(14, 3.5*batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for bi in range(batch_size):
            # Frame 0, G1
            ax = axes[bi, 0]
            hm = g1[bi, 0, 0].cpu().numpy()
            im = ax.imshow(hm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Sample {bi} - G1 Frame 0')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Middle frame, G1
            ax = axes[bi, 1]
            mid_t = g1.shape[1] // 2
            hm = g1[bi, mid_t, 0].cpu().numpy()
            im = ax.imshow(hm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Sample {bi} - G1 Frame {mid_t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Frame 0, G2
            ax = axes[bi, 2]
            hm = g2[bi, 0, 0].cpu().numpy()
            im = ax.imshow(hm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Sample {bi} - G2 Frame 0')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Middle frame, G2
            ax = axes[bi, 3]
            mid_t = g2.shape[1] // 2
            hm = g2[bi, mid_t, 0].cpu().numpy()
            im = ax.imshow(hm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Sample {bi} - G2 Frame {mid_t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle('Attention Gate Visualizations (G1/G2)\nBright=High Attention, Dark=Low Attention',
                     fontsize=14, y=0.995)
        plt.tight_layout()

        # Save
        output_path = demo_dir / 'demo_attention_heatmaps.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {output_path}")
        plt.close()

        # Also save statistics
        stats_path = demo_dir / 'demo_attention_stats.txt'
        with open(stats_path, 'w') as f:
            f.write("Attention Heatmap Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"G1 (Early Gate) - Shape: {g1.shape}\n")
            f.write(f"  Min: {g1.min():.4f}, Max: {g1.max():.4f}, Mean: {g1.mean():.4f}\n")
            f.write(f"  Std: {g1.std():.4f}\n\n")
            f.write(f"G2 (Late Gate) - Shape: {g2.shape}\n")
            f.write(f"  Min: {g2.min():.4f}, Max: {g2.max():.4f}, Mean: {g2.mean():.4f}\n")
            f.write(f"  Std: {g2.std():.4f}\n\n")
            f.write(f"HR Predictions: {hr_pred.squeeze().cpu().numpy().round(1)}\n")

        print(f"✓ Saved statistics to: {stats_path}")

        print("\nDemo complete! Check the saved images to verify attention visualization.")
        print("Good sign: Attention should show spatial structure, not uniform values.")

if __name__ == '__main__':
    demo_attention_visualization()
