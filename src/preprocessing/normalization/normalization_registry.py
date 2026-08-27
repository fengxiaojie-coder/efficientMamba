"""Registration helpers for preprocessing waveform normalization methods."""
from __future__ import annotations

from src.preprocessing.normalization.normalization_methods import (
    IdentitySignalNormalizationMethod,
    MinMaxSignalNormalizationMethod,
    RobustZSignalNormalizationMethod,
    ZScoreSignalNormalizationMethod,
)
from src.preprocessing.registry import register_preprocess_normalization


def register_default_preprocess_normalizations() -> None:
    """Register the built-in preprocessing waveform normalization methods."""
    register_preprocess_normalization(
        'none',
        'No signal normalization',
        lambda config: IdentitySignalNormalizationMethod(config),
    )
    register_preprocess_normalization(
        'zscore',
        'Z-score signal normalization',
        lambda config: ZScoreSignalNormalizationMethod(config),
    )
    register_preprocess_normalization(
        'minmax',
        'Min-max signal normalization',
        lambda config: MinMaxSignalNormalizationMethod(config),
    )
    register_preprocess_normalization(
        'robust_z',
        'Robust z-score signal normalization',
        lambda config: RobustZSignalNormalizationMethod(config),
    )