"""Reward module registry and utilities for LeRobot.

This package exposes the base :class:`RewardProvider` interface, concrete
implementations, and helpers for instantiating reward modules from
configuration objects. Reward providers produce dense reward signals and
auxiliary metadata that can be consumed by both the HIL-SERL and ConRFT
training pipelines.
"""

from .base import RewardOutput, RewardProvider
from .factory import make_reward
from .vlm import VLMProgressReward

__all__ = [
    "RewardOutput",
    "RewardProvider",
    "VLMProgressReward",
    "make_reward",
]
