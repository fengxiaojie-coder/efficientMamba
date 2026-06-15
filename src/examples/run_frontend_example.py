"""Example: run the EfficientPhys front to produce embeddings from a dummy clip.

Usage:
    python -m src.examples.run_frontend_example
"""
import torch

from src.models.efficientphys_front import EfficientPhysFront


def main():
    model = EfficientPhysFront(in_channels=6, frame_depth=16, img_size=36)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 16, 6, 36, 36)
        feats = model.forward_features(dummy)
        # Example run; avoid printing in library examples to reduce noise
        _ = tuple(dummy.shape)
        _ = tuple(feats.shape)


if __name__ == '__main__':
    main()
