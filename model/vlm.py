from __future__ import annotations

import functools
import importlib.util
import inspect
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationMixin,
    LlavaForConditionalGeneration,
    PreTrainedModel,
    ProcessorMixin,
)

from .DDPS.selection import PatchSelectionResult
from .video.selection import FrameSelectionResult, frames_to_contact_sheet, uniform_sampling
from .vqa.selection import load_image


PatchSelector = Callable[..., Any]
FrameSelector = Callable[..., torch.Tensor | FrameSelectionResult | None]
PromptInput = str | Mapping[str, Any]


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
QUANTIZED_DTYPE_KEYS = ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage")
DEFAULT_VISION_SKIP_MODULES = (
    "vision_tower",
    "vision_model",
    "visual",
    "image_tower",
)
SNAPSHOT_DOWNLOAD_KEYS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "token",
)

VLM_BACKENDS = {
    "llava": {
        "processor_cls": AutoProcessor,
        "model_cls": LlavaForConditionalGeneration,
        "model_types": {"llava"},
        "suggested_model_id": "llava-hf/llava-1.5-7b-hf",
    },
}


def _sanitize_model_id_for_path(model_id: str) -> str:
    safe = model_id.replace("\\", "/").strip()
    safe = safe.replace("/", "__").replace(":", "_").replace("@", "_")
    return "".join(
        char if (char.isalnum() or char in {"-", "_", "."}) else "_"
        for char in safe
    )


class VLMInterface(ABC):
    @abstractmethod
    def build_vlm(
        self,
        model_id: str,
    ) -> tuple[ProcessorMixin, PreTrainedModel | GenerationMixin]:
        raise NotImplementedError

    @abstractmethod
    def answer_vqa(
        self,
        prompt: PromptInput,
        *,
        image_path: str,
        **selector_kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def answer_video(
        self,
        prompt: PromptInput,
        *,
        video_path: str,
        **selector_kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def answer(
        self,
        prompt: PromptInput,
        *,
        image_path: str | None = None,
        video_path: str | None = None,
        **selector_kwargs: Any,
    ) -> str:
        if image_path and video_path:
            raise ValueError("Provide only one of `image_path` or `video_path`.")
        if image_path:
            return self.answer_vqa(
                prompt,
                image_path=image_path,
                **selector_kwargs,
            )
        if video_path:
            return self.answer_video(
                prompt,
                video_path=video_path,
                **selector_kwargs,
            )
        raise ValueError("One of `image_path` or `video_path` must be provided.")


class BaseVLM(VLMInterface):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        patch_selector: PatchSelector | None = None,
        frame_selector: FrameSelector | None = None,
        backend: str = "llava",
        image_max_side: int | None = None,
        contact_sheet_columns: int | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        dtype: str | torch.dtype | None = None,
        quantization: dict[str, Any] | None = None,
        local_model_dir: str | None = None,
        **frame_selector_kwargs: Any,
    ):
        if backend not in VLM_BACKENDS:
            available = ", ".join(sorted(VLM_BACKENDS))
            raise ValueError(f"Unsupported backend: {backend}. Available: {available}")

        backend_config = VLM_BACKENDS[backend]
        self.patch_selector = patch_selector
        self.frame_selector = frame_selector
        self.frame_selector_kwargs = frame_selector_kwargs
        self.backend = backend
        self.image_max_side = image_max_side
        self.contact_sheet_columns = contact_sheet_columns
        self.processor_cls = backend_config["processor_cls"]
        self.model_cls = backend_config["model_cls"]
        self.processor_kwargs = dict(processor_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
        self.dtype = DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
        self.quantization = dict(quantization or {})
        self.local_model_dir = (
            Path(local_model_dir).expanduser()
            if local_model_dir is not None
            else None
        )
        self.resolved_model_source = model_id
        self.last_patch_selection_info: dict[str, Any] = {
            "applied": False,
            "backend": backend,
            "reason": "not_run",
        }
        self.last_visual_selection_info: dict[str, Any] = {}
        self.last_timing_info: dict[str, Any] = {}

        self.processor: ProcessorMixin
        self.model: PreTrainedModel | GenerationMixin
        self.processor, self.model = self.build_vlm(model_id)

    def _resolve_preload_hook(self) -> tuple[Callable[..., Any], dict[str, Any]] | None:
        if self.patch_selector is None:
            return None

        selector = self.patch_selector
        selector_kwargs: dict[str, Any] = {}
        if isinstance(selector, functools.partial):
            selector_kwargs = dict(selector.keywords or {})
            selector = selector.func

        preload_hook = getattr(self.patch_selector, "preload", None)
        if preload_hook is None:
            preload_hook = getattr(selector, "preload", None)
        if preload_hook is None or not callable(preload_hook):
            return None

        return preload_hook, selector_kwargs

    def preload_runtime_resources(
        self,
        *,
        prompt: PromptInput | None = None,
        image_path: str | None = None,
        video_path: str | None = None,
    ) -> None:
        preload_target = self._resolve_preload_hook()
        if preload_target is None:
            return

        preload_hook, selector_kwargs = preload_target
        preload_inputs = {
            **selector_kwargs,
            "prompt": prompt,
            "image_path": image_path,
            "video_path": video_path,
            "processor": self.processor,
            "model": self.model,
            "backend": self.backend,
        }
        self._call_with_supported_kwargs(preload_hook, preload_inputs)

    def build_vlm(
        self,
        model_id: str,
    ) -> tuple[ProcessorMixin, PreTrainedModel | GenerationMixin]:
        use_cuda = torch.cuda.is_available()
        dtype = self.dtype or (torch.float16 if use_cuda else torch.float32)
        model_source = self._resolve_model_source(model_id)
        self._validate_backend_model_type(model_source)
        quantization_kwargs = self._build_quantization_kwargs(use_cuda=use_cuda)

        processor = self.processor_cls.from_pretrained(
            model_source,
            **self.processor_kwargs,
        )
        model_loading_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if use_cuda else None,
            **quantization_kwargs,
            **self.model_kwargs,
        }
        model = self.model_cls.from_pretrained(
            model_source,
            **model_loading_kwargs,
        )
        if not use_cuda:
            model.to("cpu")
        model.eval()
        return processor, model

    def _resolve_dtype(self, dtype_value: Any) -> Any:
        if isinstance(dtype_value, str):
            return DTYPE_MAP.get(dtype_value.lower(), dtype_value)
        return dtype_value

    def _build_quantization_kwargs(
        self,
        *,
        use_cuda: bool,
    ) -> dict[str, Any]:
        if not self.quantization or not bool(self.quantization.get("enabled", False)):
            return {}
        if "quantization_config" in self.model_kwargs:
            raise ValueError(
                "Do not set both `vlm.quantization` and `vlm.model_kwargs.quantization_config`."
            )
        if not use_cuda:
            raise RuntimeError(
                "Quantization requires CUDA. Disable `vlm.quantization.enabled` on CPU."
            )
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError("`bitsandbytes` is required for quantization.")

        mode = str(self.quantization.get("mode", "")).lower().strip()
        if mode not in {"4bit", "8bit"}:
            raise ValueError("`vlm.quantization.mode` must be one of: `4bit`, `8bit`.")

        bnb_kwargs = dict(self.quantization.get("kwargs") or {})
        for key in QUANTIZED_DTYPE_KEYS:
            if key in bnb_kwargs:
                bnb_kwargs[key] = self._resolve_dtype(bnb_kwargs[key])

        skip_modules = list(self.quantization.get("skip_modules") or [])
        existing_skip_modules = bnb_kwargs.get("llm_int8_skip_modules") or []
        if isinstance(existing_skip_modules, (list, tuple)):
            skip_modules.extend(existing_skip_modules)
        if bool(self.quantization.get("skip_vision_encoder", True)):
            skip_modules = [*skip_modules, *DEFAULT_VISION_SKIP_MODULES]
        if skip_modules:
            bnb_kwargs["llm_int8_skip_modules"] = sorted(set(skip_modules))

        bnb_kwargs["load_in_4bit"] = mode == "4bit"
        bnb_kwargs["load_in_8bit"] = mode == "8bit"
        return {"quantization_config": BitsAndBytesConfig(**bnb_kwargs)}

    def _extract_snapshot_download_kwargs(self) -> dict[str, Any]:
        download_kwargs: dict[str, Any] = {}
        for source_kwargs in (self.processor_kwargs, self.model_kwargs):
            for key in SNAPSHOT_DOWNLOAD_KEYS:
                if key in source_kwargs:
                    download_kwargs[key] = source_kwargs[key]
        return download_kwargs

    def _as_existing_local_path(self, model_id: str) -> Path | None:
        model_path = Path(model_id).expanduser()
        if model_path.exists():
            return model_path.resolve()
        return None

    def _build_local_model_path(self, model_id: str) -> Path | None:
        if self.local_model_dir is None:
            return None

        revision = self._extract_snapshot_download_kwargs().get("revision")
        revision_suffix = ""
        if revision:
            safe_revision = _sanitize_model_id_for_path(str(revision))
            revision_suffix = f"__rev_{safe_revision}"

        return self.local_model_dir / (
            f"{_sanitize_model_id_for_path(model_id)}{revision_suffix}"
        )

    def _is_local_snapshot_ready(self, local_model_path: Path) -> bool:
        return local_model_path.is_dir() and (local_model_path / "config.json").exists()

    def _resolve_model_source(self, model_id: str) -> str:
        existing_local = self._as_existing_local_path(model_id)
        if existing_local is not None:
            self.resolved_model_source = str(existing_local)
            return self.resolved_model_source

        local_model_path = self._build_local_model_path(model_id)
        if local_model_path is None:
            self.resolved_model_source = model_id
            return self.resolved_model_source

        if self._is_local_snapshot_ready(local_model_path):
            self.resolved_model_source = str(local_model_path)
            return self.resolved_model_source

        download_kwargs = self._extract_snapshot_download_kwargs()
        try:
            local_model_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_model_path),
                **download_kwargs,
            )
            if self._is_local_snapshot_ready(local_model_path):
                self.resolved_model_source = str(local_model_path)
                return self.resolved_model_source
        except Exception as exc:
            warnings.warn(
                "Failed to mirror model snapshot into local_model_dir; "
                f"falling back to default Hugging Face loading path. reason={exc}",
                stacklevel=2,
            )

        self.resolved_model_source = model_id
        return self.resolved_model_source

    def _extract_config_load_kwargs(self) -> dict[str, Any]:
        config_kwargs: dict[str, Any] = {}
        for source_kwargs in (self.processor_kwargs, self.model_kwargs):
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "revision",
                "subfolder",
                "token",
                "trust_remote_code",
            ):
                if key in source_kwargs:
                    config_kwargs[key] = source_kwargs[key]
        return config_kwargs

    def _validate_backend_model_type(self, model_id: str) -> None:
        backend_config = VLM_BACKENDS[self.backend]
        expected_model_types = backend_config.get("model_types")
        if not expected_model_types:
            return

        try:
            config = AutoConfig.from_pretrained(
                model_id,
                **self._extract_config_load_kwargs(),
            )
        except Exception:
            return

        model_type = str(getattr(config, "model_type", "")).lower()
        if not model_type or model_type in expected_model_types:
            return

        expected_display = ", ".join(sorted(expected_model_types))
        suggestion = backend_config.get("suggested_model_id")
        raise ValueError(
            f"Incompatible backend/model pair: backend `{self.backend}` expects "
            f"`config.model_type` in {{{expected_display}}}, but `{model_id}` has "
            f"`config.model_type={model_type}`. For example, use `{suggestion}`."
        )

    def _normalize_prompt_input(self, prompt: PromptInput) -> tuple[str, str]:
        if isinstance(prompt, Mapping):
            system_prompt = str(prompt.get("system", "") or "").strip()
            user_prompt = str(prompt.get("user", "") or "").strip()
        else:
            system_prompt = ""
            user_prompt = str(prompt).strip()

        if not user_prompt:
            raise ValueError("Prompt must include a non-empty user prompt.")
        return system_prompt, user_prompt

    def _build_chat_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        has_image: bool,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            user_prompt = f"{system_prompt}\n\n{user_prompt}"

        user_content: list[dict[str, Any]] = []
        if has_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _prepare_text_input(self, prompt: PromptInput, *, has_image: bool) -> str:
        system_prompt, user_prompt = self._normalize_prompt_input(prompt)
        if hasattr(self.processor, "apply_chat_template"):
            messages = self._build_chat_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                has_image=has_image,
            )
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_text = user_prompt
        if system_prompt:
            prompt_text = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
        if has_image and "<image>" not in prompt_text:
            prompt_text = f"<image>\nUSER: {prompt_text}\nASSISTANT:"
        return prompt_text

    def _load_vqa_input(self, *, image_path: str) -> tuple[Image.Image, dict[str, Any]]:
        image_selection = load_image(image_path, max_side=self.image_max_side)
        return image_selection.image, {
            "type": "image",
            **image_selection.metadata,
        }

    def _load_video_input(
        self,
        *,
        video_path: str,
        selector_kwargs: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        frame_selector = self.frame_selector or uniform_sampling
        frame_selection = self._normalize_frame_selection_output(
            frame_selector(
                video_path=video_path,
                **{**self.frame_selector_kwargs, **selector_kwargs},
            ),
            video_path=video_path,
        )
        image = frames_to_contact_sheet(
            frame_selection,
            columns=self.contact_sheet_columns,
            annotate=True,
        )
        frame_count = int(frame_selection.frames.shape[0]) if frame_selection.frames is not None else 0
        contact_sheet_columns = self.contact_sheet_columns
        if contact_sheet_columns is None and frame_count > 0:
            import math

            contact_sheet_columns = int(math.ceil(math.sqrt(frame_count)))
        contact_sheet_rows = None
        if contact_sheet_columns is not None and frame_count > 0:
            import math

            contact_sheet_rows = int(math.ceil(frame_count / max(int(contact_sheet_columns), 1)))
        return image, {
            "type": "video_contact_sheet",
            **frame_selection.metadata,
            "contact_sheet_size": list(image.size),
            "contact_sheet_columns": contact_sheet_columns,
            "contact_sheet_rows": contact_sheet_rows,
            "contact_sheet_label_height": 18,
            "_frame_selection": frame_selection,
        }

    def _normalize_frame_selection_output(
        self,
        selection_output: torch.Tensor | FrameSelectionResult | None,
        *,
        video_path: str,
    ) -> FrameSelectionResult:
        if selection_output is None:
            return FrameSelectionResult(
                frames=None,
                metadata={"video_path": video_path, "num_frames": 0},
            )
        if isinstance(selection_output, FrameSelectionResult):
            metadata = dict(selection_output.metadata)
            metadata.setdefault("video_path", video_path)
            return FrameSelectionResult(
                frames=selection_output.frames,
                metadata=metadata,
            )
        if torch.is_tensor(selection_output):
            return FrameSelectionResult(
                frames=selection_output,
                metadata={
                    "video_path": video_path,
                    "num_frames": int(selection_output.shape[0]),
                },
            )
        raise TypeError(
            "Frame selector output must be a torch.Tensor, FrameSelectionResult, or None."
        )

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_model_inputs(
        self,
        prompt_text: str,
        image: Image.Image | None,
    ) -> dict[str, Any]:
        processor_inputs: dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
        }
        if image is not None:
            processor_inputs["images"] = image

        inputs = self.processor(**processor_inputs)
        device = self._model_device()
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _decode_generation_output(
        self,
        output_ids: torch.Tensor,
        prompt_length: int,
    ) -> str:
        if output_ids.ndim == 2 and output_ids.shape[1] >= prompt_length:
            output_ids = output_ids[:, prompt_length:]
        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

    def _count_generated_tokens(
        self,
        output_ids: torch.Tensor,
        prompt_length: int,
    ) -> int:
        if output_ids.ndim != 2:
            return 0
        return max(int(output_ids.shape[1]) - int(prompt_length), 0)

    def _run_standard_generation(self, model_inputs: dict[str, Any]) -> str:
        generate_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                **self.generation_kwargs,
            )
        generate_elapsed = time.perf_counter() - generate_start

        input_ids = model_inputs.get("input_ids")
        prompt_length = input_ids.shape[1] if input_ids is not None else 0
        generated_tokens = self._count_generated_tokens(
            output_ids,
            prompt_length=prompt_length,
        )
        self.last_timing_info = {
            "path": "standard_generation",
            "llm_generate_seconds": generate_elapsed,
            "generated_tokens": generated_tokens,
        }
        return self._decode_generation_output(output_ids, prompt_length=prompt_length)

    def _call_with_supported_kwargs(
        self,
        fn: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> Any:
        signature = inspect.signature(fn)
        accepts_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_keyword:
            return fn(**kwargs)
        filtered_kwargs = {
            name: value for name, value in kwargs.items() if name in signature.parameters
        }
        return fn(**filtered_kwargs)

    def _extract_image_features(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("Image inputs are required for patch selection.")

        get_image_features = getattr(self.model, "get_image_features", None)
        if callable(get_image_features):
            feature_kwargs = {
                "pixel_values": pixel_values,
                "vision_feature_layer": getattr(self.model.config, "vision_feature_layer", -2),
                "vision_feature_select_strategy": getattr(
                    self.model.config,
                    "vision_feature_select_strategy",
                    "default",
                ),
            }
            image_features = self._call_with_supported_kwargs(
                get_image_features,
                feature_kwargs,
            )
        else:
            vision_tower = getattr(self.model, "vision_tower", None)
            projector = getattr(self.model, "multi_modal_projector", None)
            if vision_tower is None or projector is None:
                raise RuntimeError(
                    f"Backend `{self.backend}` does not expose LLaVA image features."
                )
            vision_outputs = vision_tower(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            layer = int(getattr(self.model.config, "vision_feature_layer", -2))
            selected_features = vision_outputs.hidden_states[layer]
            strategy = str(
                getattr(self.model.config, "vision_feature_select_strategy", "default")
            )
            if strategy == "default":
                selected_features = selected_features[:, 1:]
            image_features = projector(selected_features)

        if isinstance(image_features, (list, tuple)):
            if len(image_features) != 1:
                raise ValueError("Patch selection currently expects a single image.")
            image_features = image_features[0]
        if image_features.ndim == 3:
            if image_features.shape[0] != 1:
                raise ValueError("Patch selection currently expects a single image.")
            image_features = image_features[0]
        if image_features.ndim != 2:
            raise ValueError(
                "Expected image features with shape (N, D), "
                f"but got {tuple(image_features.shape)}."
            )
        return image_features, {"image_token_count": int(image_features.shape[0])}

    def _call_patch_selector(
        self,
        image_features: torch.Tensor,
        *,
        prompt: PromptInput,
        image: Image.Image,
        model_inputs: dict[str, Any],
        extraction_metadata: dict[str, Any],
        visual_metadata: dict[str, Any],
    ) -> Any:
        if self.patch_selector is None:
            return None

        selector_kwargs = {
            "image_features": image_features,
            "prompt": prompt,
            "image": image,
            "model_inputs": model_inputs,
            "extraction_metadata": extraction_metadata,
            "visual_metadata": visual_metadata,
            "frame_selection": visual_metadata.get("_frame_selection"),
            "processor": self.processor,
            "model": self.model,
            "backend": self.backend,
        }
        return self._call_with_supported_kwargs(self.patch_selector, selector_kwargs)

    def _coerce_patch_indices(
        self,
        indices: torch.Tensor | list[int] | tuple[int, ...],
        *,
        device: torch.device,
        upper_bound: int,
    ) -> torch.Tensor:
        tensor = (
            indices
            if torch.is_tensor(indices)
            else torch.tensor(indices, device=device, dtype=torch.long)
        )
        tensor = tensor.to(device=device, dtype=torch.long).flatten()
        if torch.any(tensor < 0) or torch.any(tensor >= upper_bound):
            raise ValueError(
                f"Patch selector indices must be within [0, {upper_bound}), got {tensor.tolist()}."
            )
        return tensor.unique(sorted=True)

    def _normalize_patch_selection_output(
        self,
        selection_output: Any,
        full_image_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        selected_indices: torch.Tensor | None = None
        selected_features: torch.Tensor | None = None
        metadata: dict[str, Any] = {}

        if selection_output is None:
            selected_indices = torch.arange(
                full_image_features.shape[0],
                device=full_image_features.device,
                dtype=torch.long,
            )
            selected_features = full_image_features
            return selected_indices, selected_features, metadata

        if isinstance(selection_output, PatchSelectionResult):
            selected_indices = selection_output.selected_indices
            selected_features = selection_output.selected_features
            metadata = dict(selection_output.metadata)
        elif isinstance(selection_output, dict):
            selected_indices = selection_output.get("selected_indices")
            selected_features = selection_output.get("selected_features")
            metadata = {
                key: value
                for key, value in selection_output.items()
                if key not in {"selected_indices", "selected_features"}
            }
        elif torch.is_tensor(selection_output):
            if selection_output.ndim == 1:
                selected_indices = selection_output
            elif selection_output.ndim == 2:
                selected_features = selection_output
            else:
                raise ValueError(
                    "Tensor patch selector output must be 1D indices or 2D features."
                )
        elif isinstance(selection_output, Sequence) and not isinstance(
            selection_output,
            (str, bytes),
        ) and all(isinstance(item, int) for item in selection_output):
            selected_indices = torch.tensor(
                selection_output,
                device=full_image_features.device,
                dtype=torch.long,
            )
        else:
            raise TypeError(
                "Unsupported patch selector output type. Expected None, "
                "PatchSelectionResult, dict, Tensor, or list[int]."
            )

        if selected_indices is None and selected_features is None:
            raise ValueError(
                "Patch selector must return `selected_indices`, `selected_features`, or both."
            )

        if selected_indices is not None:
            selected_indices = self._coerce_patch_indices(
                selected_indices,
                device=full_image_features.device,
                upper_bound=full_image_features.shape[0],
            )
        if selected_features is None:
            selected_features = full_image_features[selected_indices]
        else:
            selected_features = selected_features.to(
                device=full_image_features.device,
                dtype=full_image_features.dtype,
            )
        if selected_indices is None:
            selected_indices = torch.arange(
                selected_features.shape[0],
                device=full_image_features.device,
                dtype=torch.long,
            )
        if selected_features.ndim != 2:
            raise ValueError(
                "Selected image features must have shape (N, D), "
                f"but got {tuple(selected_features.shape)}."
            )
        if selected_indices.numel() != selected_features.shape[0]:
            raise ValueError(
                "Patch selector returned mismatched indices/features: "
                f"indices={selected_indices.numel()}, features={selected_features.shape[0]}."
            )
        return selected_indices, selected_features, metadata

    def _build_generation_inputs_from_patch_selection(
        self,
        model_inputs: dict[str, Any],
        full_image_features: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_features: torch.Tensor,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids is None:
            raise ValueError("`input_ids` is required for patch selection.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        image_token_id = getattr(self.model.config, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Could not find the image placeholder token id.")

        image_positions = torch.nonzero(
            input_ids[0] == int(image_token_id),
            as_tuple=False,
        ).flatten()
        if int(image_positions.numel()) != int(full_image_features.shape[0]):
            return None, {
                "reason": "image_placeholder_feature_mismatch",
                "image_placeholders": int(image_positions.numel()),
                "image_features": int(full_image_features.shape[0]),
            }

        kept_positions = image_positions[selected_indices]
        keep_mask = torch.ones(
            input_ids.shape[1],
            dtype=torch.bool,
            device=input_ids.device,
        )
        keep_mask[image_positions] = False
        keep_mask[kept_positions] = True

        pruned_input_ids = input_ids[:, keep_mask]
        pruned_attention_mask = attention_mask[:, keep_mask]
        pruned_inputs_embeds = self.model.get_input_embeddings()(pruned_input_ids)

        pruned_image_mask = pruned_input_ids[0] == int(image_token_id)
        pruned_image_features = selected_features.to(
            device=pruned_inputs_embeds.device,
            dtype=pruned_inputs_embeds.dtype,
        )
        if int(pruned_image_mask.sum().item()) != int(pruned_image_features.shape[0]):
            raise ValueError(
                "Pruned image placeholder count does not match selected feature count: "
                f"tokens={int(pruned_image_mask.sum().item())}, "
                f"features={int(pruned_image_features.shape[0])}."
            )

        pruned_inputs_embeds[0, pruned_image_mask] = pruned_image_features
        position_ids = pruned_attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(pruned_attention_mask == 0, 0)

        generation_inputs = {
            "input_ids": pruned_input_ids,
            "inputs_embeds": pruned_inputs_embeds,
            "attention_mask": pruned_attention_mask,
            "position_ids": position_ids,
        }
        metadata = {
            "original_image_tokens": int(full_image_features.shape[0]),
            "selected_image_tokens": int(pruned_image_features.shape[0]),
            "input_length_before": int(input_ids.shape[1]),
            "input_length_after": int(pruned_input_ids.shape[1]),
        }
        return generation_inputs, metadata

    @torch.inference_mode()
    def _run_patch_selection_generation(
        self,
        *,
        prompt: PromptInput,
        image: Image.Image,
        model_inputs: dict[str, Any],
        visual_metadata: dict[str, Any],
    ) -> str:
        full_image_features, extraction_metadata = self._extract_image_features(model_inputs)
        selection_output = self._call_patch_selector(
            image_features=full_image_features,
            prompt=prompt,
            image=image,
            model_inputs=model_inputs,
            extraction_metadata=extraction_metadata,
            visual_metadata=visual_metadata,
        )
        selected_indices, selected_features, selector_metadata = (
            self._normalize_patch_selection_output(
                selection_output=selection_output,
                full_image_features=full_image_features,
            )
        )
        generation_inputs, pruning_metadata = self._build_generation_inputs_from_patch_selection(
            model_inputs=model_inputs,
            full_image_features=full_image_features,
            selected_indices=selected_indices,
            selected_features=selected_features,
        )
        selector_runtime_metadata = {
            key: selector_metadata[key]
            for key in (
                "original_video_tokens",
                "selected_video_tokens",
                "reallocated_token_count",
            )
            if key in selector_metadata
        }
        if generation_inputs is None:
            self.last_patch_selection_info = {
                "applied": False,
                "backend": self.backend,
                **extraction_metadata,
                **selector_runtime_metadata,
                "selector_metadata": selector_metadata,
                **pruning_metadata,
            }
            return self._run_standard_generation(model_inputs)

        generate_start = time.perf_counter()
        output_ids = self.model.generate(
            **generation_inputs,
            **self.generation_kwargs,
        )
        generate_elapsed = time.perf_counter() - generate_start

        prompt_length = generation_inputs["input_ids"].shape[1]
        generated_tokens = self._count_generated_tokens(
            output_ids,
            prompt_length=prompt_length,
        )
        self.last_patch_selection_info = {
            "applied": True,
            "backend": self.backend,
            **extraction_metadata,
            **pruning_metadata,
            **selector_runtime_metadata,
            "selector_metadata": selector_metadata,
            "selector_output_keys": sorted(selector_metadata.keys()),
        }
        self.last_timing_info = {
            "path": "patch_selection_generation",
            "llm_generate_seconds": generate_elapsed,
            "generated_tokens": generated_tokens,
        }
        return self._decode_generation_output(output_ids, prompt_length=prompt_length)

    def _answer_with_visual(
        self,
        prompt: PromptInput,
        *,
        image: Image.Image,
        visual_metadata: dict[str, Any],
    ) -> str:
        self.last_visual_selection_info = {
            key: value for key, value in visual_metadata.items() if not key.startswith("_")
        }
        prompt_text = self._prepare_text_input(prompt, has_image=True)
        model_inputs = self._build_model_inputs(prompt_text=prompt_text, image=image)
        self.last_timing_info = {}

        if self.patch_selector is not None:
            return self._run_patch_selection_generation(
                prompt=prompt,
                image=image,
                model_inputs=model_inputs,
                visual_metadata=visual_metadata,
            )

        self.last_patch_selection_info = {
            "applied": False,
            "backend": self.backend,
            "reason": "patch_selector_not_configured",
        }
        return self._run_standard_generation(model_inputs)

    def answer_vqa(
        self,
        prompt: PromptInput,
        *,
        image_path: str,
        **selector_kwargs: Any,
    ) -> str:
        del selector_kwargs
        image, visual_metadata = self._load_vqa_input(image_path=image_path)
        return self._answer_with_visual(
            prompt,
            image=image,
            visual_metadata=visual_metadata,
        )

    def answer_video(
        self,
        prompt: PromptInput,
        *,
        video_path: str,
        **selector_kwargs: Any,
    ) -> str:
        image, visual_metadata = self._load_video_input(
            video_path=video_path,
            selector_kwargs=selector_kwargs,
        )
        return self._answer_with_visual(
            prompt,
            image=image,
            visual_metadata=visual_metadata,
        )

    def answer(
        self,
        prompt: PromptInput,
        *,
        image_path: str | None = None,
        video_path: str | None = None,
        **selector_kwargs: Any,
    ) -> str:
        if image_path and video_path:
            raise ValueError("Provide only one of `image_path` or `video_path`.")
        if image_path:
            return self.answer_vqa(
                prompt,
                image_path=image_path,
                **selector_kwargs,
            )
        if video_path:
            return self.answer_video(
                prompt,
                video_path=video_path,
                **selector_kwargs,
            )
        raise ValueError("One of `image_path` or `video_path` must be provided.")
