"""Configuration objects for reward providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus


@dataclass
class RewardConfig(draccus.ChoiceRegistry):
    """Base class for reward provider configuration."""

    goal: str | None = None
    device: str | None = None
    reward_mode: Literal["delta", "abs"] = "delta"
    dense_lambda: float = 0.8
    success_threshold: float = 0.95
    allow_sparse_fallback: bool = True

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        cli_overrides: list[str] | None = None,
    ) -> RewardConfig:
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "reward_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Reward config not found at {config_path}")

        with draccus.config_type("json"):
            return draccus.parse(cls, str(config_path), args=cli_overrides or [])


@RewardConfig.register_subclass("vlm_progress")
@dataclass
class VLMProgressRewardConfig(RewardConfig):
    """Configuration for the SmolVLM2-inspired dense reward."""

    model_name: str | None = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    window_size: int = 6
    num_shuffles: int = 4
    ema_beta: float = 0.9
    milestones_path: str | None = None
    exemplar_paths: list[str] = field(default_factory=list)
    generate_text_explanations: bool = False
    freeze_backbone: bool = True
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    torch_dtype: str | None = None
