"""Registration helpers for preprocessing presets, stages, and pipelines."""
from __future__ import annotations

from typing import Optional

from src.preprocessing.registry import (
    register_preprocess_pipeline,
    register_preprocess_preset,
    register_preprocess_stage,
)


def register_default_preprocess_entries() -> None:
    """Register the default UBFC preprocessing preset, stages, and pipeline."""
    from src.preprocessing.pipelines.ubfc_pipeline import (
        NormalizationStage,
        PreprocessConfig,
        PreprocessPipeline,
        ROIStage,
        SamplingStage,
    )

    def build_default_preprocess_config() -> PreprocessConfig:
        return PreprocessConfig()

    def build_default_preprocess_pipeline(config: Optional[PreprocessConfig] = None):
        return PreprocessPipeline(config)

    register_preprocess_preset(
        'ubfc_default',
        'Current UBFC preprocessing default settings',
        build_default_preprocess_config,
    )
    register_preprocess_stage('roi', 'ROI extraction stage', lambda config: ROIStage(config))
    register_preprocess_stage(
        'normalization',
        'Signal normalization stage',
        lambda config: NormalizationStage(config),
    )
    register_preprocess_stage(
        'sampling',
        'Sliding-window sampling stage',
        lambda config: SamplingStage(config),
    )
    register_preprocess_pipeline(
        'ubfc_default',
        'Default UBFC preprocessing pipeline',
        build_default_preprocess_pipeline,
    )
