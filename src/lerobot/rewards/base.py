"""Base interfaces for dense reward providers."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class RewardOutput:
    """Container for dense reward predictions.

    Attributes:
        progress: Smoothed task progress estimate in ``[0, 1]``.
        reward: Dense reward value in ``[0, 1]``.
        milestones: Optional mapping of milestone names to boolean states.
        text_explanation: Optional textual rationale produced by the model.
        extras: Additional provider-specific metadata that should be logged but
            not relied upon for control.
    """

    progress: float
    reward: float
    milestones: Mapping[str, bool] | None = None
    text_explanation: str | None = None
    extras: MutableMapping[str, Any] | None = None


class RewardProvider(Protocol):
    """Protocol for reward providers used by LeRobot."""

    def reset(self) -> None:
        """Reset internal state at the beginning of a new episode."""

    def compute(
        self,
        observation: Mapping[str, Any],
        *,
        info: MutableMapping[str, Any] | None = None,
    ) -> RewardOutput:
        """Compute a dense reward signal for the given observation."""

    def close(self) -> None:
        """Clean up any allocated resources."""


def ensure_reward_output(value: RewardOutput | Mapping[str, Any]) -> RewardOutput:
    """Coerce legacy mapping outputs into :class:`RewardOutput`.

    Reward providers may return dictionaries for convenience. This helper makes
    it easy to support both styles.
    """

    if isinstance(value, RewardOutput):
        return value

    return RewardOutput(
        progress=float(value.get("progress", 0.0)),
        reward=float(value.get("reward", 0.0)),
        milestones=value.get("milestones"),
        text_explanation=value.get("text_explanation"),
        extras=value.get("extras"),
    )
