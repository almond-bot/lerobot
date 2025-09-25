import pytest
import torch

from lerobot.configs.reward import VLMProgressRewardConfig
from lerobot.rewards.base import RewardOutput
from lerobot.rewards.vlm import VLMProgressReward
from lerobot.scripts.rl.learner import collect_vlm_metrics, merge_with_corrections
from lerobot.utils.buffer import ReplayBuffer


class DummyCorrectionsBuffer:
    def __init__(self, batch):
        self._batch = batch

    def sample(self, batch_size):
        return self._batch

    def __len__(self):
        return 1


def make_obs(intensity: float) -> dict[str, torch.Tensor]:
    value = torch.full((3, 8, 8), intensity, dtype=torch.float32)
    return {"observation.image_primary": value}


def test_vlm_progress_reward_delta_and_abs_modes():
    cfg = VLMProgressRewardConfig(
        window_size=2,
        num_shuffles=1,
        ema_beta=0.0,
        reward_mode="delta",
        goal="test",
        model_name=None,
    )
    reward = VLMProgressReward.from_config(cfg)

    obs_sequence = [make_obs(v) for v in (0.0, 64.0, 128.0, 255.0)]

    deltas = []
    for obs in obs_sequence:
        output = reward.compute(obs)
        deltas.append(output.reward)

    assert deltas[0] == 0.0
    assert deltas[-1] > deltas[1] > 0.0

    cfg_abs = VLMProgressRewardConfig(
        window_size=1,
        num_shuffles=1,
        ema_beta=0.0,
        reward_mode="abs",
        goal="test",
        model_name=None,
    )
    reward_abs = VLMProgressReward.from_config(cfg_abs)
    abs_values = []
    progress_values = []
    for obs in obs_sequence:
        output = reward_abs.compute(obs)
        abs_values.append(output.reward)
        progress_values.append(output.progress)

    assert abs_values[0] == pytest.approx(0.0)
    assert abs_values[-1] >= abs_values[-2] >= abs_values[1]
    assert abs_values == pytest.approx(progress_values)


def test_merge_with_corrections_prioritizes_hil_transitions():
    base_batch = {
        "state": {"obs": torch.zeros(1, 1)},
        "action": torch.zeros(1, 1),
        "reward": torch.tensor([0.1], dtype=torch.float32),
        "next_state": {"obs": torch.zeros(1, 1)},
        "done": torch.tensor([0.0], dtype=torch.float32),
        "truncated": torch.tensor([0.0], dtype=torch.float32),
        "complementary_info": {"is_intervention": torch.tensor([0])},
    }

    corrections_batch = {
        "state": {"obs": torch.ones(1, 1)},
        "action": torch.ones(1, 1),
        "reward": torch.tensor([0.9], dtype=torch.float32),
        "next_state": {"obs": torch.ones(1, 1)},
        "done": torch.tensor([0.0], dtype=torch.float32),
        "truncated": torch.tensor([0.0], dtype=torch.float32),
        "complementary_info": {"is_intervention": torch.tensor([1])},
    }

    merged = merge_with_corrections(base_batch, DummyCorrectionsBuffer(corrections_batch), batch_size=4)
    assert merged["reward"][0].item() == pytest.approx(0.9)
    assert merged["complementary_info"]["is_intervention"][0].item() == 1


def test_dataset_conversion_handles_reward_provider_errors():
    class FlakyProvider:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def compute(self, observation, *, info=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("intentional failure")
            if info is not None:
                info["progress_call"] = self.calls
            return RewardOutput(
                progress=0.75,
                reward=0.5,
                milestones={"reach": True},
                extras={"prev_progress": 0.5},
            )

        def close(self):  # pragma: no cover - interface requirement only
            pass

    class DummyDataset:
        def __init__(self):
            self.frames = [
                {
                    "observation.image_primary": torch.zeros(3, 4, 4),
                    "action": torch.zeros(1),
                    "next.reward": torch.tensor([0.0]),
                    "next.done": torch.tensor([False]),
                    "episode_index": torch.tensor([0]),
                },
                {
                    "observation.image_primary": torch.ones(3, 4, 4),
                    "action": torch.ones(1),
                    "next.reward": torch.tensor([0.0]),
                    "next.done": torch.tensor([True]),
                    "episode_index": torch.tensor([0]),
                },
            ]

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):
            return self.frames[idx]

    dataset = DummyDataset()
    transitions = ReplayBuffer._lerobotdataset_to_transitions(
        dataset=dataset,
        state_keys=["observation.image_primary"],
        reward_provider=FlakyProvider(),
    )

    assert len(transitions) == len(dataset)

    first_info = transitions[0]["complementary_info"]
    assert first_info is not None
    assert torch.allclose(first_info["vlm_progress"], torch.tensor([0.0]))
    assert torch.allclose(first_info["vlm_reward"], torch.tensor([0.0]))
    assert first_info["vlm_milestone_reach"].item() == 0

    second_info = transitions[1]["complementary_info"]
    assert second_info is not None
    assert torch.allclose(second_info["vlm_progress"], torch.tensor([0.75]))
    assert torch.allclose(second_info["vlm_prev_progress"], torch.tensor([0.5]))
    assert torch.allclose(second_info["vlm_reward"], torch.tensor([0.5]))
    assert second_info["vlm_milestone_reach"].item() == 1


def test_collect_vlm_metrics_aggregates_signals():
    complementary_info = {
        "vlm_progress": torch.tensor([0.2, 0.6, 0.8]),
        "vlm_prev_progress": torch.tensor([0.1, 0.4, 0.7]),
        "vlm_reward": torch.tensor([0.05, 0.2, 0.4]),
        "vlm_milestone_grasp": torch.tensor([1, 0, 1], dtype=torch.int32),
        "vlm_milestone_place": torch.tensor([0, 1, 1], dtype=torch.int32),
    }

    metrics = collect_vlm_metrics(complementary_info)

    assert metrics["vlm_progress_mean"] == pytest.approx(0.5333, rel=1e-3)
    assert metrics["vlm_prev_progress_mean"] == pytest.approx(0.4, rel=1e-3)
    assert metrics["vlm_reward_mean"] == pytest.approx(0.2166, rel=1e-3)
    assert metrics["vlm_milestone_mean"] == pytest.approx(0.6666, rel=1e-3)
