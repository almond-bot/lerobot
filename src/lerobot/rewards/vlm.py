"""Zero/few-shot VLM dense reward provider."""

from __future__ import annotations

import logging
import random
import re
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:  # Optional dependency – degrade gracefully when unavailable
    import torch
except Exception:  # pragma: no cover - torch should exist during real training
    torch = None  # type: ignore

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - pillow is expected but optional
    raise ImportError("Pillow is required for VLM rewards") from exc

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except Exception:  # pragma: no cover - optional dependency
    AutoModelForVision2Seq = None  # type: ignore
    AutoProcessor = None  # type: ignore

try:  # Optional LoRA support
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

from lerobot.configs.reward import RewardConfig, VLMProgressRewardConfig

from .base import RewardOutput, RewardProvider

LOGGER = logging.getLogger(__name__)

_PROGRESS_PROMPT = (
    "Task: {goal}. From this image, estimate completion in [0,1]. "
    "Return ONLY a number with two decimals. Consider approach, grasp, "
    "alignment, insertion/placement, verification. If not started: 0.00; "
    "if complete and stable: 1.00."
)

_MILESTONE_PROMPT = "Task: {goal}. From this image, is `{milestone}` achieved? Return 'yes' or 'no' only."


@dataclass
class _FrameRecord:
    idx: int
    image: Image.Image
    raw_progress: float | None = None


class VLMProgressReward(RewardProvider):
    """SmolVLM2-inspired dense reward with sliding-window smoothing."""

    def __init__(
        self,
        *,
        goal: str,
        model_name: str | None,
        device: str,
        window_size: int,
        num_shuffles: int,
        ema_beta: float,
        reward_mode: str,
        success_threshold: float,
        milestones_path: str | None,
        exemplar_paths: Iterable[str] | None,
        generate_text: bool,
        freeze_backbone: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: float,
        torch_dtype: str | None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if reward_mode not in {"delta", "abs"}:
            raise ValueError("reward_mode must be 'delta' or 'abs'")

        self.goal = goal or ""
        self.model_name = model_name
        self.device = device
        self.window_size = window_size
        self.num_shuffles = max(1, num_shuffles)
        self.ema_beta = float(np.clip(ema_beta, 0.0, 0.999))
        self.reward_mode = reward_mode
        self.success_threshold = float(np.clip(success_threshold, 0.0, 1.0))
        self.generate_text = generate_text

        self._window: deque[_FrameRecord] = deque(maxlen=window_size)
        self._frame_counter = 0
        self._ema_progress: float | None = None
        self._prev_progress = 0.0
        self._prev_raw_progress = 0.0

        self._milestone_prompts = self._load_milestones(milestones_path)
        self._exemplar_snippets = self._load_exemplars(exemplar_paths)

        self._processor = None
        self._model = None
        self._torch_dtype = self._parse_dtype(torch_dtype)
        self._load_backbone(
            freeze_backbone=freeze_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        self.reset()

    # ------------------------------------------------------------------ utils
    @classmethod
    def from_config(cls, cfg: RewardConfig, **kwargs: Any) -> VLMProgressReward:
        if not isinstance(cfg, VLMProgressRewardConfig):
            raise TypeError(f"Expected VLMProgressRewardConfig, got {type(cfg)}")

        goal = cfg.goal or ""
        device = cfg.device or kwargs.get("device", "cpu")

        return cls(
            goal=goal,
            model_name=cfg.model_name,
            device=device,
            window_size=cfg.window_size,
            num_shuffles=cfg.num_shuffles,
            ema_beta=cfg.ema_beta,
            reward_mode=cfg.reward_mode,
            success_threshold=cfg.success_threshold,
            milestones_path=cfg.milestones_path,
            exemplar_paths=cfg.exemplar_paths,
            generate_text=cfg.generate_text_explanations,
            freeze_backbone=cfg.freeze_backbone,
            use_lora=cfg.use_lora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            torch_dtype=cfg.torch_dtype,
        )

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        self._window.clear()
        self._frame_counter = 0
        self._ema_progress = None
        self._prev_progress = 0.0
        self._prev_raw_progress = 0.0

    def close(self) -> None:  # pragma: no cover - close not used in tests
        self._window.clear()
        self._model = None
        self._processor = None

    # ----------------------------------------------------------------- compute
    def compute(
        self,
        observation: Mapping[str, Any],
        *,
        info: MutableMapping[str, Any] | None = None,
    ) -> RewardOutput:
        image = self._extract_image(observation)
        record = _FrameRecord(idx=self._frame_counter, image=image)
        self._frame_counter += 1
        self._window.append(record)

        raw_progress = self._predict_window_progress()
        smoothed = self._apply_ema(raw_progress)

        reward = smoothed
        if self.reward_mode == "delta":
            reward = max(smoothed - self._prev_progress, 0.0)
        reward = float(np.clip(reward, 0.0, 1.0))

        milestones = self._evaluate_milestones(record.image, smoothed)
        explanation = self._maybe_generate_text(record.image, smoothed)

        extras: MutableMapping[str, Any] = {
            "raw_progress": raw_progress,
            "ema_progress": smoothed,
            "prev_progress": self._prev_progress,
        }

        if info is not None:
            info.setdefault("vlm_progress_raw", raw_progress)
            info.setdefault("vlm_progress", smoothed)
            info.setdefault("vlm_reward", reward)
            info.setdefault("vlm_prev_progress", self._prev_progress)
            info.setdefault("vlm_milestones", milestones or {})
            if explanation:
                info.setdefault("vlm_text", explanation)

        self._prev_progress = smoothed
        self._prev_raw_progress = raw_progress

        return RewardOutput(
            progress=smoothed,
            reward=reward,
            milestones=milestones,
            text_explanation=explanation,
            extras=extras,
        )

    @property
    def milestone_names(self) -> tuple[str, ...]:
        """Names of configured milestone checks."""

        return tuple(self._milestone_prompts.keys())

    # --------------------------------------------------------------- prediction
    def _predict_window_progress(self) -> float:
        if not self._window:
            return self._prev_raw_progress

        idxs = list(range(len(self._window)))
        predictions: list[float] = []
        for _ in range(self.num_shuffles):
            order = idxs.copy()
            random.shuffle(order)
            ordered = [self._window[i] for i in order]
            seq_preds = self._predict_sequence(ordered)
            original_position = order.index(len(self._window) - 1)
            predictions.append(seq_preds[original_position])

        try:
            raw = float(np.clip(median(predictions), 0.0, 1.0))
        except Exception:
            raw = self._prev_raw_progress
        return raw

    def _predict_sequence(self, records: Iterable[_FrameRecord]) -> list[float]:
        outputs: list[float] = []
        for record in records:
            if record.raw_progress is None:
                record.raw_progress = self._predict_single(record.image)
            outputs.append(float(np.clip(record.raw_progress, 0.0, 1.0)))
        return outputs

    def _predict_single(self, image: Image.Image) -> float:
        if self._model is None or AutoModelForVision2Seq is None:
            return self._heuristic_progress(image)

        try:
            assert torch is not None
            assert self._processor is not None
            device = torch.device(self.device)
            prompt = self._build_prompt()
            inputs = self._prepare_inputs(prompt, image, device=device)
            with torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=8)
            text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            return self._parse_progress(text)
        except Exception as exc:  # pragma: no cover - depends on external models
            LOGGER.debug("Falling back to heuristic progress due to %s", exc)
            return self._heuristic_progress(image)

    # -------------------------------------------------------------- heuristics
    def _heuristic_progress(self, image: Image.Image) -> float:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        if arr.size == 0:
            return 0.0
        normalized = arr / 255.0
        return float(np.clip(normalized.mean(), 0.0, 1.0))

    def _apply_ema(self, value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        if self._ema_progress is None:
            self._ema_progress = value
        else:
            self._ema_progress = self.ema_beta * self._ema_progress + (1 - self.ema_beta) * value
        return float(np.clip(self._ema_progress, 0.0, 1.0))

    def _evaluate_milestones(self, image: Image.Image, progress: float) -> dict[str, bool] | None:
        if not self._milestone_prompts:
            return None

        results: dict[str, bool] = {}
        for name, prompt in self._milestone_prompts.items():
            if self._model is None:
                # Use progress-based heuristic when VLM unavailable
                results[name] = progress >= self.success_threshold
                continue
            try:
                assert torch is not None
                assert self._processor is not None
                device = torch.device(self.device)
                text_prompt = prompt
                if "{" in prompt:
                    text_prompt = prompt.format(goal=self.goal, milestone=name)
                else:
                    milestone_text = prompt or name
                    text_prompt = _MILESTONE_PROMPT.format(goal=self.goal, milestone=milestone_text)
                inputs = self._prepare_inputs(text_prompt, image, device=device)
                with torch.inference_mode():
                    generated = self._model.generate(**inputs, max_new_tokens=4)
                text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
                results[name] = self._parse_yes_no(text)
            except Exception as exc:  # pragma: no cover - depends on external models
                LOGGER.debug("Milestone '%s' fallback due to %s", name, exc)
                results[name] = progress >= self.success_threshold
        return results

    def _maybe_generate_text(self, image: Image.Image, progress: float) -> str | None:
        if not self.generate_text:
            return None
        if self._model is None:
            return f"Heuristic progress {progress:.2f}"
        return f"Estimated progress {progress:.2f}"

    # ---------------------------------------------------------- helper methods
    def _extract_image(self, observation: Mapping[str, Any]) -> Image.Image:
        candidate = None
        for key, value in observation.items():
            if "image" in key:
                candidate = value
                if "primary" in key:
                    break
        if candidate is None:
            raise KeyError("Observation does not contain an image key")

        if isinstance(candidate, Image.Image):
            return candidate
        if torch is not None and isinstance(candidate, torch.Tensor):
            arr = candidate.detach().cpu().numpy()
        else:
            arr = np.asarray(candidate)

        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _build_prompt(self) -> str:
        exemplar_str = ""
        if self._exemplar_snippets:
            exemplar_str = "\n".join(self._exemplar_snippets)
        base = _PROGRESS_PROMPT.format(goal=self.goal)
        if exemplar_str:
            base = f"{exemplar_str}\n{base}"
        return base

    def _prepare_inputs(
        self,
        prompt: str,
        image: Image.Image,
        *,
        device: torch.device,
    ) -> dict[str, Any]:
        assert torch is not None
        assert self._processor is not None

        formatted_prompt = self._format_prompt_for_processor(prompt)
        inputs = self._processor(
            text=[formatted_prompt],
            images=[image],
            return_tensors="pt",
        )

        tensor_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                tensor_inputs[key] = value.to(device)
            else:
                tensor_inputs[key] = value
        return tensor_inputs

    def _format_prompt_for_processor(self, prompt: str) -> str:
        if self._processor is None:
            return prompt

        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]
            try:
                return apply_chat_template(conversation, add_generation_prompt=True)
            except TypeError:
                # Older versions of transformers may not support keyword args
                return apply_chat_template(conversation)
        return prompt

    @staticmethod
    def _parse_progress(text: str) -> float:
        match = re.search(r"([01]?\.\d+|[01])", text)
        if not match:
            return 0.0
        value = float(match.group(1))
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        text = text.strip().lower()
        if text.startswith("y"):
            return True
        if text.startswith("n"):
            return False
        # fall back to keyword search
        return "yes" in text and "no" not in text

    def _load_backbone(
        self,
        *,
        freeze_backbone: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: float,
    ) -> None:
        if self.model_name is None or AutoModelForVision2Seq is None or AutoProcessor is None:
            LOGGER.info("Using heuristic reward model (no transformers backend)")
            return

        try:
            assert torch is not None
            dtype = self._torch_dtype or torch.float16
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device)

            if freeze_backbone:
                for param in self._model.parameters():
                    param.requires_grad = False

            if use_lora and get_peft_model is not None and LoraConfig is not None:
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules="all",  # rely on PEFT defaults
                )
                self._model = get_peft_model(self._model, lora_config)
        except Exception as exc:  # pragma: no cover - depends on external assets
            LOGGER.warning("Failed to load VLM backbone %s: %s", self.model_name, exc)
            self._model = None
            self._processor = None

    @staticmethod
    def _load_milestones(path: str | None) -> dict[str, str]:
        if not path:
            return {}
        if yaml is None:
            LOGGER.warning("PyYAML not available; ignoring milestones file %s", path)
            return {}
        try:
            with Path(path).expanduser().open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, Mapping):
                raise ValueError("Milestone file must map names to prompts")
            return {str(k): str(v) for k, v in data.items()}
        except Exception as exc:
            LOGGER.warning("Failed to load milestones from %s: %s", path, exc)
            return {}

    @staticmethod
    def _load_exemplars(paths: Iterable[str] | None) -> list[str]:
        snippets: list[str] = []
        if not paths:
            return snippets
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.exists():
                LOGGER.warning("Exemplar path %s missing", path)
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    snippets.append(text)
            except Exception as exc:
                LOGGER.warning("Failed to read exemplar %s: %s", path, exc)
        return snippets

    @staticmethod
    def _parse_dtype(dtype: str | None):
        if dtype is None or torch is None:
            return None
        name = dtype.lower()
        if hasattr(torch, name):
            return getattr(torch, name)
        LOGGER.warning("Unknown torch dtype %s; defaulting to float16", dtype)
        return torch.float16
