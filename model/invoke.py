from __future__ import annotations

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from model.vlm import VLMInterface


def load_prompt(invoke_config: DictConfig) -> dict[str, str]:
    prompt_file = invoke_config.get("prompt_file")
    if not prompt_file:
        raise ValueError("`invoke.prompt_file` must be provided in the config.")

    prompt_path = Path(to_absolute_path(str(prompt_file)))
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    query_file = invoke_config.get("query_file")
    if not query_file:
        raise ValueError("`invoke.query_file` must be provided in the config.")

    query_path = Path(to_absolute_path(str(query_file)))
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    query = query_path.read_text(encoding="utf-8").strip()
    if not query:
        raise ValueError(f"Query file is empty: {query_path}")

    sections = {"system": [], "user": []}
    current_section: str | None = None
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[SYSTEM]":
            current_section = "system"
            continue
        if stripped == "[USER]":
            current_section = "user"
            continue
        if current_section is not None:
            sections[current_section].append(line)

    system_prompt = "\n".join(sections["system"]).strip()
    user_prompt = "\n".join(sections["user"]).strip()
    if not user_prompt:
        raise ValueError(f"`[USER]` section is missing in prompt file: {prompt_path}")

    user_prompt = user_prompt.format(query=str(query))
    return {
        "system": system_prompt,
        "user": user_prompt,
        "query": query,
    }


def resolve_inference_task(invoke_config: DictConfig | None) -> str:
    task = "auto" if invoke_config is None else str(invoke_config.get("task", "auto"))
    normalized = task.strip().lower()
    aliases = {
        "image": "vqa",
        "vqa": "vqa",
        "video": "video",
        "auto": "auto",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        available = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported `invoke.task`: {task}. Available: {available}.")
    return resolved


def _select_patch_selector_config(config: DictConfig, inference_task: str) -> Any:
    if inference_task == "auto":
        return config.vlm.get("patch_selector", config.get("patch_selection"))

    missing = object()
    nested_selector = OmegaConf.select(
        config,
        f"patch_selection.{inference_task}",
        default=missing,
    )
    if nested_selector is not missing:
        return nested_selector

    task_selector_key = f"patch_selection_{inference_task}"
    if task_selector_key in config:
        return config.get(task_selector_key)
    return config.get("patch_selection")


def _configure_vlm_for_task(config: DictConfig, inference_task: str) -> None:
    if inference_task == "auto":
        return

    patch_selector_cfg = _select_patch_selector_config(config, inference_task)
    with open_dict(config):
        config.vlm.patch_selector = patch_selector_cfg


def build_vlm(config: DictConfig, *, inference_task: str = "auto") -> VLMInterface:
    _configure_vlm_for_task(config, inference_task)
    vlm = instantiate(config.vlm)
    if not isinstance(vlm, VLMInterface):
        raise TypeError("Instantiated `vlm` does not implement VLMInterface.")
    return vlm


@contextmanager
def suppress_model_loading_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    logger_names = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "transformers",
    )
    previous_levels = {name: logging.getLogger(name).level for name in logger_names}
    progress_bars_were_disabled = are_progress_bars_disabled()

    disable_progress_bars()
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
        if not progress_bars_were_disabled:
            enable_progress_bars()


def _resolve_existing_path(value: Any, *, label: str) -> Path:
    path = Path(to_absolute_path(str(value)))
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def resolve_visual_path(invoke_config: DictConfig) -> tuple[str, Path, str]:
    inference_task = resolve_inference_task(invoke_config)
    image_value = invoke_config.get("image_path")
    video_value = invoke_config.get("video_path")

    if inference_task == "vqa":
        if video_value:
            raise ValueError("`invoke.task: vqa` expects `invoke.image_path`, not `invoke.video_path`.")
        if not image_value:
            raise ValueError("`invoke.task: vqa` requires `invoke.image_path`.")
        return "image_path", _resolve_existing_path(image_value, label="Image"), "vqa"

    if inference_task == "video":
        if image_value:
            raise ValueError("`invoke.task: video` expects `invoke.video_path`, not `invoke.image_path`.")
        if not video_value:
            raise ValueError("`invoke.task: video` requires `invoke.video_path`.")
        return "video_path", _resolve_existing_path(video_value, label="Video"), "video"

    if image_value and video_value:
        raise ValueError("Provide only one of `invoke.image_path` or `invoke.video_path`.")
    if image_value:
        return "image_path", _resolve_existing_path(image_value, label="Image"), "vqa"
    if video_value:
        return "video_path", _resolve_existing_path(video_value, label="Video"), "video"
    raise ValueError("One of `invoke.image_path` or `invoke.video_path` must be provided.")


def summarize_config(config: DictConfig, *, inference_task: str) -> dict[str, Any]:
    patch_selection_cfg = _select_patch_selector_config(config, inference_task)
    patch_selection = (
        OmegaConf.to_container(patch_selection_cfg, resolve=True)
        if patch_selection_cfg is not None
        else None
    )
    frame_selection_cfg = config.get("frame_selection") if inference_task == "video" else None
    frame_selection = (
        OmegaConf.to_container(frame_selection_cfg, resolve=True)
        if frame_selection_cfg is not None
        else None
    )
    generation_kwargs = OmegaConf.to_container(
        config.vlm.get("generation_kwargs"),
        resolve=True,
    )
    invoke_config = OmegaConf.to_container(config.invoke, resolve=True)

    return {
        "inference_task": inference_task,
        "frame_selection": frame_selection,
        "patch_selection": patch_selection,
        "invoke": invoke_config,
        "model_id": config.vlm.get("model_id"),
        "backend": config.vlm.get("backend"),
        "dtype": config.vlm.get("dtype"),
        "local_model_dir": config.vlm.get("local_model_dir"),
        "inference_batch_size": config.vlm.get("inference_batch_size"),
        "generation_kwargs": generation_kwargs,
    }


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig) -> None:
    invoke_config = config.get("invoke")
    if invoke_config is None:
        raise ValueError("`invoke` section must be provided in the config.")

    visual_arg_name, visual_path, inference_task = resolve_visual_path(invoke_config)
    prompt = load_prompt(invoke_config)
    with suppress_model_loading_output(
        enabled=invoke_config.get("quiet_model_loading", True),
    ):
        vlm = build_vlm(config, inference_task=inference_task)
        preload_runtime_resources = getattr(vlm, "preload_runtime_resources", None)
        if callable(preload_runtime_resources):
            preload_runtime_resources(
                **{visual_arg_name: str(visual_path)},
                prompt=prompt,
            )

    if invoke_config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(config, resolve=True).strip())
        print()

    summary = summarize_config(config, inference_task=inference_task)
    print("=== Inference Setup ===")
    print(f"Input       : {visual_path}")
    print(f"Task        : {summary['inference_task']}")
    print(f"Input Type  : {visual_arg_name.removesuffix('_path')}")
    print(f"Model       : {summary['model_id']}")
    print(f"Backend     : {summary['backend']}")
    print(f"DType       : {summary['dtype']}")
    if summary.get("local_model_dir") is not None:
        print(f"Local Store : {summary['local_model_dir']}")
    resolved_model_source = getattr(vlm, "resolved_model_source", None)
    if resolved_model_source and str(resolved_model_source) != str(summary["model_id"]):
        print(f"Load Source : {resolved_model_source}")
    frame_selection = summary.get("frame_selection")
    if frame_selection is not None:
        print(f"Video Layer : {frame_selection['_target_']}")
    patch_selection = summary.get("patch_selection")
    if patch_selection is not None:
        print(f"Patch Layer : {patch_selection['_target_']}")
    print(f"Prompt File : {summary['invoke']['prompt_file']}")
    print(f"Query File  : {summary['invoke']['query_file']}")
    print()

    start_time = time.perf_counter()
    answer_method = vlm.answer_vqa if inference_task == "vqa" else vlm.answer_video
    response = answer_method(
        prompt=prompt,
        **{visual_arg_name: str(visual_path)},
    )
    elapsed = time.perf_counter() - start_time

    print("=== Response ===")
    print(response.strip())
    print()
    print("=== Timing ===")
    print(f"Elapsed     : {elapsed:.2f}s")
    timing_info = getattr(vlm, "last_timing_info", {}) or {}
    llm_generate_seconds = timing_info.get("llm_generate_seconds")
    generated_tokens = timing_info.get("generated_tokens")
    if llm_generate_seconds is not None:
        llm_generate_seconds = float(llm_generate_seconds)
        print(f"LLM Generate: {llm_generate_seconds:.2f}s")
        print(f"Non-LLM     : {max(elapsed - llm_generate_seconds, 0.0):.2f}s")
        if generated_tokens is not None:
            generated_tokens = int(generated_tokens)
            print(f"Gen Tokens  : {generated_tokens}")
            if llm_generate_seconds > 0:
                print(f"LLM Tok/s   : {generated_tokens / llm_generate_seconds:.2f}")


if __name__ == "__main__":
    main()
