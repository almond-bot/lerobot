"""Factory utilities for reward providers."""

from __future__ import annotations

import logging
from typing import Any

from lerobot.configs.reward import RewardConfig

from .base import RewardProvider
from .vlm import VLMProgressReward

LOGGER = logging.getLogger(__name__)


def make_reward(cfg: RewardConfig | None, **kwargs: Any) -> RewardProvider | None:
    """Instantiate a reward provider from a configuration object."""

    if cfg is None:
        return None

    if cfg.type == "vlm_progress":
        return VLMProgressReward.from_config(cfg, **kwargs)

    LOGGER.warning("Unknown reward config type %s; returning None", cfg.type)
    return None
