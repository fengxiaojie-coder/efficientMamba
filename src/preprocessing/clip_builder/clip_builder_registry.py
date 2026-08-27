"""Registration helpers for preprocessing clip-construction methods."""
from __future__ import annotations

from src.preprocessing.clip_builder.clip_builders import (
    DiffOnlyClipBuilder,
    MeanCenteredClipBuilder,
    RawOnlyClipBuilder,
    RawPlusDiffClipBuilder,
)
from src.preprocessing.registry import register_preprocess_clip_builder


def register_default_preprocess_clip_builders() -> None:
    """Register the built-in preprocessing clip-construction methods."""
    register_preprocess_clip_builder(
        'raw_plus_diff',
        'Concatenate temporal differences and raw RGB frames',
        lambda config: RawPlusDiffClipBuilder(config),
    )
    register_preprocess_clip_builder(
        'raw_only',
        'Use only raw RGB frames in each clip',
        lambda config: RawOnlyClipBuilder(config),
    )
    register_preprocess_clip_builder(
        'diff_only',
        'Use only temporal difference frames in each clip',
        lambda config: DiffOnlyClipBuilder(config),
    )
    register_preprocess_clip_builder(
        'mean_centered',
        'Subtract the per-frame mean from each window before clipping',
        lambda config: MeanCenteredClipBuilder(config),
    )