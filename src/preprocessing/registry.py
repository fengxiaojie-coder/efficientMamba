"""Central preprocessing pipeline and stage registries for preprocessing entrypoints.

Add new preprocessing workflows or stage implementations here and they become
available to the CLI without additional code changes.
"""
from __future__ import annotations

from typing import Any, Callable

PREPROCESS_PIPELINE_REGISTRY: dict[str, dict[str, Any]] = {}
PREPROCESS_STAGE_REGISTRY: dict[str, dict[str, Any]] = {}
PREPROCESS_PRESET_REGISTRY: dict[str, dict[str, Any]] = {}
ROI_STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {}
PREPROCESS_NORMALIZATION_REGISTRY: dict[str, dict[str, Any]] = {}
PREPROCESS_CLIP_BUILDER_REGISTRY: dict[str, dict[str, Any]] = {}


def classify_preprocess_components() -> dict[str, list[tuple[str, str]]]:
    """Return a categorized view of the registered preprocessing components."""
    return {
        'pipelines': [
            (name, spec['display_name'])
            for name, spec in sorted(PREPROCESS_PIPELINE_REGISTRY.items())
        ],
        'presets': [
            (name, spec['display_name'])
            for name, spec in sorted(PREPROCESS_PRESET_REGISTRY.items())
        ],
        'stages': [
            (name, spec['display_name'])
            for name, spec in sorted(PREPROCESS_STAGE_REGISTRY.items())
        ],
        'roi_strategies': [
            (name, spec['display_name'])
            for name, spec in sorted(ROI_STRATEGY_REGISTRY.items())
        ],
        'normalizations': [
            (name, spec['display_name'])
            for name, spec in sorted(PREPROCESS_NORMALIZATION_REGISTRY.items())
        ],
        'clip_builders': [
            (name, spec['display_name'])
            for name, spec in sorted(PREPROCESS_CLIP_BUILDER_REGISTRY.items())
        ],
    }


def register_preprocess_pipeline(name: str, display_name: str, builder: Callable[[Any], Any]) -> None:
    PREPROCESS_PIPELINE_REGISTRY[name] = {
        'display_name': display_name,
        'builder': builder,
    }


def list_preprocess_pipelines() -> list[str]:
    return list(PREPROCESS_PIPELINE_REGISTRY.keys())


def build_preprocess_pipeline(name: str, config: Any):
    if name not in PREPROCESS_PIPELINE_REGISTRY:
        raise ValueError(
            f'Unknown preprocess pipeline: {name}. Available: {list_preprocess_pipelines()}'
        )
    spec = PREPROCESS_PIPELINE_REGISTRY[name]
    return spec['builder'](config)


def register_preprocess_preset(name: str, display_name: str, builder: Callable[[], Any]) -> None:
    PREPROCESS_PRESET_REGISTRY[name] = {
        'display_name': display_name,
        'builder': builder,
    }


def list_preprocess_presets() -> list[str]:
    return list(PREPROCESS_PRESET_REGISTRY.keys())


def build_preprocess_preset(name: str):
    if name not in PREPROCESS_PRESET_REGISTRY:
        raise ValueError(
            f'Unknown preprocess preset: {name}. Available: {list_preprocess_presets()}'
        )
    spec = PREPROCESS_PRESET_REGISTRY[name]
    return spec['builder']()


def register_preprocess_stage(name: str, display_name: str, builder: Callable[[Any], Any]) -> None:
    PREPROCESS_STAGE_REGISTRY[name] = {
        'display_name': display_name,
        'builder': builder,
    }


def list_preprocess_stages() -> list[str]:
    return list(PREPROCESS_STAGE_REGISTRY.keys())


def build_preprocess_stage(name: str, config: Any):
    if name not in PREPROCESS_STAGE_REGISTRY:
        raise ValueError(
            f'Unknown preprocess stage: {name}. Available: {list_preprocess_stages()}'
        )
    spec = PREPROCESS_STAGE_REGISTRY[name]
    return spec['builder'](config)


def register_roi_strategy(name: str, display_name: str, builder: Callable[[], Any]) -> None:
    ROI_STRATEGY_REGISTRY[name] = {
        'display_name': display_name,
        'builder': builder,
    }


def list_roi_strategies() -> list[str]:
    return list(ROI_STRATEGY_REGISTRY.keys())


def build_roi_strategy(name: str):
    if name not in ROI_STRATEGY_REGISTRY:
        raise ValueError(
            f'Unknown ROI strategy: {name}. Available: {list_roi_strategies()}'
        )
    spec = ROI_STRATEGY_REGISTRY[name]
    return spec['builder']()


def register_preprocess_normalization(name: str, display_name: str, builder: Callable[[Any], Any]) -> None:
    PREPROCESS_NORMALIZATION_REGISTRY[name] = {
        'display_name': display_name,
        'builder': builder,
    }


def list_preprocess_normalizations() -> list[str]:
    return list(PREPROCESS_NORMALIZATION_REGISTRY.keys())


def build_preprocess_normalization(name: str, config: Any):
    if name not in PREPROCESS_NORMALIZATION_REGISTRY:
        raise ValueError(
            f'Unknown preprocess normalization: {name}. Available: {list_preprocess_normalizations()}'
        )
    spec = PREPROCESS_NORMALIZATION_REGISTRY[name]
    return spec['builder'](config)


def register_preprocess_clip_builder(name: str, display_name: str, builder: Callable[[Any], Any]) -> None:
    PREPROCESS_CLIP_BUILDER_REGISTRY[name] = {
        'display_name': display_name,
        'builder': builder,
    }


def list_preprocess_clip_builders() -> list[str]:
    return list(PREPROCESS_CLIP_BUILDER_REGISTRY.keys())


def build_preprocess_clip_builder(name: str, config: Any):
    if name not in PREPROCESS_CLIP_BUILDER_REGISTRY:
        raise ValueError(
            f'Unknown preprocess clip builder: {name}. Available: {list_preprocess_clip_builders()}'
        )
    spec = PREPROCESS_CLIP_BUILDER_REGISTRY[name]
    return spec['builder'](config)
