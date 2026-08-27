"""Central model registry and factory for train/eval entrypoints.

Add new architectures here, then train.py/eval.py can pick them up
automatically without additional code changes.
"""
from __future__ import annotations

from typing import Any

from .efficientphys_baseline import EfficientPhysBaselineRegressor
from .efficientphys_mamba import EfficientPhysMambaRegressor, mamba_backend_status
from .physformer_baseline import PhysFormerBaselineRegressor
from .rhythmmamba_baseline import RhythmMambaBaselineRegressor


# Keep insertion order for predictable CLI help and default listing.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'efficientphys': {
        'display_name': 'EfficientPhysBaselineRegressor',
        'builder': lambda frame_depth, temporal_backbone='auto': EfficientPhysBaselineRegressor(
            in_channels=6,
            frame_depth=frame_depth,
        ),
    },
    'physformer': {
        'display_name': 'PhysFormerBaselineRegressor',
        'builder': lambda frame_depth, temporal_backbone='auto': PhysFormerBaselineRegressor(
            frame_depth=frame_depth,
        ),
    },
    'efficientphys_mamba': {
        'display_name': 'EfficientPhysMambaRegressor',
        'builder': lambda frame_depth, temporal_backbone='auto': EfficientPhysMambaRegressor(
            in_channels=6,
            frame_depth=frame_depth,
            use_mamba=(temporal_backbone != 'gru'),
        ),
    },
    'rhythmmamba': {
        'display_name': 'RhythmMambaBaselineRegressor',
        'builder': lambda frame_depth, temporal_backbone='auto': RhythmMambaBaselineRegressor(
            frame_depth=frame_depth,
            in_channels=6,
        ),
    },
}


def list_model_arches() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def build_model(model_arch: str, frame_depth: int, temporal_backbone: str = 'auto'):
    if model_arch not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model_arch: {model_arch}. Available: {list_model_arches()}')

    spec = MODEL_REGISTRY[model_arch]
    model = spec['builder'](frame_depth=frame_depth, temporal_backbone=temporal_backbone)

    mamba_available, mamba_reason = mamba_backend_status()
    if model_arch == 'efficientphys_mamba' and temporal_backbone == 'mamba' and not model.temporal.uses_mamba:
        raise SystemExit(
            'Requested --temporal_backbone mamba, but Mamba backend is unavailable. '
            f'Reason: {mamba_reason}'
        )

    total_params = sum(p.numel() for p in model.parameters())
    if model_arch == 'efficientphys_mamba':
        temporal_params = sum(p.numel() for p in model.temporal.parameters())
        backbone_name = 'Mamba' if model.temporal.uses_mamba else 'GRU'
    elif model_arch == 'physformer':
        temporal_params = total_params
        backbone_name = 'PhysFormer'
    elif model_arch == 'rhythmmamba':
        temporal_params = total_params
        backbone_name = 'RhythmMamba'
    else:
        temporal_params = total_params
        backbone_name = 'EfficientPhys'

    model_info = {
        'model_arch': model_arch,
        'display_name': spec['display_name'],
        'backbone_name': backbone_name,
        'total_params': total_params,
        'temporal_params': temporal_params,
        'mamba_available': mamba_available,
        'mamba_reason': mamba_reason,
    }
    return model, model_info
