from __future__ import annotations

import argparse
import html
import inspect
import math
import sys
import tempfile
import time
import webbrowser
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
HTML_SUFFIXES = {".html", ".htm"}
DATASET_CHOICES = ("direct", "nextqa", "mvbench", "videomme", "egoschema")
DEFAULT_DIRECT_VIDEO_PATH = REPO_ROOT / "data" / "12223108496.mp4"
DEFAULT_DIRECT_QUERY_TEXT = "The color of the wall."
DEFAULT_PATCH_SELECTION_KEY = "patch_selection_v4"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.invoke import load_prompt
from model.PatchSelection.DenseDPS.selection_v1 import (
    _compute_sliding_window_merged_scores,
    _compute_window_score_maps,
    _expand_scores_for_frame_duplication,
    _load_queries,
    _prepare_frame_arrays,
    _resolve_clip_dtype,
    _resolve_device_key,
    _resolve_patch_scoring_frames,
    _resolve_selection_mode,
    _select_topk_per_frame,
)
from model.PatchSelection.DenseDPS.selection_v2 import (
    _load_maskclip_components_v2,
    _load_text_embeddings_v2,
    _resolve_total_budget,
)
from model.PatchSelection.DenseDPS.selection_v4 import (
    _allocate_budget_with_softmax_capacities,
    _compute_eligible_patch_mask,
    _select_topk_from_eligible_patches,
)
from model.PatchSelection.DenseDPS.selection_v5 import (
    _add_gaussian_noise_to_query_embeddings,
    _compute_dense_patch_score_maps_and_clean_frame_scores,
)
from model.video.selection import FrameSelectionResult


@dataclass(slots=True)
class VisualizationTarget:
    video_path: Path
    prompt: dict[str, str]
    query_text: str
    sample_label: str
    query_file: Path | None = None


@dataclass(slots=True)
class SelectionDiagnostics:
    frame_selection: FrameSelectionResult
    selected_indices: torch.Tensor
    selector_metadata: dict[str, Any]
    extraction_metadata: dict[str, Any]
    query_scores: torch.Tensor | None = None
    query_winners: torch.Tensor | None = None
    selected_mask: torch.Tensor | None = None
    raw_clip_selected_mask: torch.Tensor | None = None
    raw_clip_query_scores: torch.Tensor | None = None
    raw_clip_query_winners: torch.Tensor | None = None
    raw_clip_metadata: dict[str, Any] | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="선택/비선택 비디오 패치를 Plotly HTML로 시각화합니다.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "DDPS.yaml",
        help="실험 config 파일 경로",
    )
    parser.add_argument(
        "--patch-selection-key",
        type=str,
        default=DEFAULT_PATCH_SELECTION_KEY,
        help="config 내 patch_selection_v1/v2/... 같은 selector key를 직접 선택",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="direct",
        help="직접 비디오를 지정하거나 지원 eval dataset에서 sample을 선택",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="dataset sample index")
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="dataset sample id/qid/video id 등으로 sample 선택",
    )
    parser.add_argument("--task-name", type=str, default=None, help="MVBench task filter")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_DIRECT_VIDEO_PATH,
        help="직접 사용할 비디오 경로",
    )
    parser.add_argument("--query-file", type=Path, default=None, help="직접 사용할 query file")
    parser.add_argument(
        "--query-text",
        type=str,
        default=None,
        help="query를 문자열로 직접 지정",
    )
    parser.add_argument("--prompt-file", type=Path, default=None, help="prompt template override")
    parser.add_argument("--dataset-root", type=Path, default=None, help="dataset root override")
    parser.add_argument("--questions-file", type=Path, default=None, help="question file override")
    parser.add_argument("--annotation-file", type=Path, default=None, help="annotation file override")
    parser.add_argument("--annotations-dir", type=Path, default=None, help="annotation dir override")
    parser.add_argument("--videos-dir", type=Path, default=None, help="video dir override")
    parser.add_argument("--data-dir", type=Path, default=None, help="MVBench data dir override")
    parser.add_argument("--uid-map-file", type=Path, default=None, help="EgoSchema uid map override")
    parser.add_argument("--video-map-file", type=Path, default=None, help="Video-MME video map override")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test" / "artifacts" / "selected_patch.html",
        help="Plotly HTML 출력 경로",
    )
    parser.add_argument(
        "--max-frame-panels",
        type=int,
        default=24,
        help="HTML에 직접 펼칠 최대 temporal frame panel 수",
    )
    parser.add_argument(
        "--panel-cols",
        type=int,
        default=4,
        help="frame panel column 수",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=None,
        help="Dense patch scoring 시 한 번에 GPU로 올릴 frame 수. 지정하지 않으면 config 값을 사용합니다.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="patch_score_threshold를 시각화 실행에서만 override",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="budget allocation temperature를 시각화 실행에서만 override",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="patch keep_ratio를 시각화 실행에서만 override",
    )
    parser.add_argument(
        "--skip-query-diagnostics",
        action="store_true",
        help="query별 patch score/winner 재계산을 생략",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="진행 상황 로그를 출력하지 않음",
    )
    parser.add_argument("--show", action="store_true", help="저장 후 browser로 열기")
    return parser.parse_args()


def _progress(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "quiet_progress", False):
        print(f"[show_selected_patch] {message}", flush=True)


def _load_plot_backend() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit("plotly가 필요합니다. 예: `pip install plotly`") from exc
    return go, make_subplots


def _to_abs_path(path_value: str | Path | None) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(to_absolute_path(str(path_value))).expanduser()
    return path.resolve()


def _resolve_output_path(output_path: Path) -> tuple[Path, bool]:
    resolved = output_path.expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    coerced = resolved.suffix.lower() not in HTML_SUFFIXES
    if coerced:
        resolved = resolved.with_suffix(".html")
    return resolved, coerced


def _apply_common_config_overrides(config: DictConfig, args: argparse.Namespace) -> None:
    if args.prompt_file is not None:
        config.invoke.prompt_file = str(_to_abs_path(args.prompt_file))
    if args.patch_selection_key is not None:
        selector_cfg = OmegaConf.select(config, args.patch_selection_key)
        if selector_cfg is None:
            raise KeyError(f"config에서 patch selector key를 찾을 수 없습니다: {args.patch_selection_key}")
        config.vlm.patch_selector = selector_cfg
    if args.clip_batch_size is not None and args.clip_batch_size <= 0:
        raise ValueError(f"`--clip-batch-size` must be positive, got {args.clip_batch_size}.")
    if args.threshold is not None and args.threshold < 0.0:
        raise ValueError(f"`--threshold` must be non-negative, got {args.threshold}.")
    if args.temperature is not None and args.temperature <= 0.0:
        raise ValueError(f"`--temperature` must be positive, got {args.temperature}.")
    if args.keep_ratio is not None and not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError(f"`--keep-ratio` must be in (0, 1], got {args.keep_ratio}.")
    selector = config.vlm.get("patch_selector")
    if args.clip_batch_size is not None and selector is not None and "batch_size" in selector:
        selector.batch_size = int(args.clip_batch_size)
    if args.threshold is not None and selector is not None:
        if "patch_score_threshold" not in selector:
            raise KeyError("선택한 patch selector에는 patch_score_threshold 옵션이 없습니다.")
        selector.patch_score_threshold = float(args.threshold)
    if args.temperature is not None and selector is not None:
        if "temperature" not in selector:
            raise KeyError("선택한 patch selector에는 temperature 옵션이 없습니다.")
        selector.temperature = float(args.temperature)
    if args.keep_ratio is not None and selector is not None:
        if "keep_ratio" not in selector:
            raise KeyError("선택한 patch selector에는 keep_ratio 옵션이 없습니다.")
        selector.keep_ratio = float(args.keep_ratio)


def _apply_dataset_overrides(eval_config: DictConfig, args: argparse.Namespace) -> None:
    override_pairs = {
        "dataset_root": args.dataset_root,
        "questions_file": args.questions_file,
        "annotation_file": args.annotation_file,
        "annotations_dir": args.annotations_dir,
        "videos_dir": args.videos_dir,
        "data_dir": args.data_dir,
        "uid_map_file": args.uid_map_file,
        "video_map_file": args.video_map_file,
    }
    for key, value in override_pairs.items():
        if value is not None:
            eval_config[key] = str(_to_abs_path(value))
    if args.task_name:
        eval_config["task_names"] = [args.task_name]


def _sample_matches_id(sample: Any, sample_id: str) -> bool:
    target = str(sample_id).strip().casefold()
    candidate_names = (
        "q_uid",
        "question_id",
        "video_id",
        "video_name",
        "video",
        "task_name",
    )
    for name in candidate_names:
        value = getattr(sample, name, None)
        if value is not None and str(value).strip().casefold() == target:
            return True
    raw_item = getattr(sample, "raw_item", None)
    if isinstance(raw_item, dict):
        return any(str(value).strip().casefold() == target for value in raw_item.values())
    return False


def _pick_sample(samples: list[Any], args: argparse.Namespace) -> Any:
    if not samples:
        raise ValueError("선택 가능한 dataset sample이 없습니다.")
    if args.sample_id:
        for sample in samples:
            if _sample_matches_id(sample, args.sample_id):
                return sample
        raise ValueError(f"sample id와 일치하는 항목이 없습니다: {args.sample_id}")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(
            f"sample index 범위를 벗어났습니다: {args.sample_index} / {len(samples)}"
        )
    return samples[args.sample_index]


def _dataset_eval_config(config: DictConfig, dataset: str) -> DictConfig:
    key = "egoschema" if dataset == "egoschema" else dataset
    existing = config.get(key)
    if existing is None:
        return OmegaConf.create({})
    return OmegaConf.create(OmegaConf.to_container(existing, resolve=True))


def _prompt_file_from_config(config: DictConfig) -> Path:
    return _to_abs_path(config.invoke.prompt_file) or (REPO_ROOT / "model" / "base" / "prompt.txt")


def _load_dataset_target(
    *,
    config: DictConfig,
    args: argparse.Namespace,
    temp_dir: Path,
) -> VisualizationTarget:
    eval_config = _dataset_eval_config(config, args.dataset)
    _apply_dataset_overrides(eval_config, args)
    prompt_file = _prompt_file_from_config(config)

    if args.dataset == "nextqa":
        import eval.nextqa as nextqa

        dataset_root, questions_file, videos_dir = nextqa._resolve_dataset_layout(eval_config)
        del dataset_root
        samples = nextqa._load_samples(questions_file=questions_file, videos_dir=videos_dir)
        sample = _pick_sample(samples, args)
        prompt = nextqa._render_prompt(
            prompt_template=nextqa._load_prompt_template(prompt_file),
            sample=sample,
        )
        query_text = nextqa._normalize_query_text(sample.question)
        sample_label = f"NextQA {sample.q_uid}"
        video_path = sample.video_path
    elif args.dataset == "mvbench":
        import eval.mvbench as mvbench

        dataset_root, annotations_dir, data_dir = mvbench._resolve_dataset_layout(eval_config)
        del dataset_root
        task_names = [args.task_name] if args.task_name else list(eval_config.get("task_names") or [])
        annotation_files = mvbench._collect_annotation_files(annotations_dir, task_names or None)
        video_lookup, duplicate_keys = mvbench._build_video_lookup(data_dir)
        samples, _, _ = mvbench._load_samples(
            annotation_files=annotation_files,
            video_lookup=video_lookup,
            duplicate_keys=duplicate_keys,
        )
        sample = _pick_sample(samples, args)
        prompt = mvbench._render_prompt(
            prompt_template=mvbench._load_prompt_template(prompt_file),
            sample=sample,
        )
        query_text = sample.question
        clip_context = nullcontext(sample.video_path)
        if sample.start is not None or sample.end is not None:
            clip_context = nullcontext(
                mvbench._extract_video_clip(sample.video_path, sample.start, sample.end, temp_dir)
            )
        with clip_context as clip_path:
            video_path = Path(clip_path)
        sample_label = f"MVBench {sample.task_name}/{sample.sample_index}"
    elif args.dataset == "videomme":
        import eval.videomme as videomme

        dataset_root, annotation_file, videos_dir, video_map_file = videomme._resolve_dataset_layout(eval_config)
        del dataset_root
        indexed_videos = videomme._index_videos(videos_dir)
        video_lookup = videomme._build_video_lookup(indexed_videos)
        video_map = videomme._load_video_map(video_map_file)
        samples, _ = videomme._load_samples(
            annotation_file=annotation_file,
            indexed_videos=indexed_videos,
            video_lookup=video_lookup,
            video_map=video_map,
        )
        sample = _pick_sample(samples, args)
        prompt = videomme._render_prompt(
            prompt_template=videomme._load_prompt_template(prompt_file),
            sample=sample,
        )
        query_text = videomme._normalize_query_text(sample.question)
        sample_label = f"Video-MME {sample.video_id}/{sample.question_id}"
        video_path = sample.video_path
    elif args.dataset == "egoschema":
        import eval.ego as ego

        dataset_root, questions_file, uid_map_file, videos_dir = ego._resolve_dataset_layout(eval_config)
        del dataset_root
        samples = ego._load_samples(
            questions_file=questions_file,
            uid_map_file=uid_map_file,
            videos_dir=videos_dir,
        )
        sample = _pick_sample(samples, args)
        prompt = ego._render_prompt(
            prompt_template=ego._load_prompt_template(prompt_file),
            sample=sample,
        )
        query_text = ego._normalize_query_text(sample.question)
        sample_label = f"EgoSchema {sample.q_uid}"
        video_path = sample.video_path
    else:
        raise ValueError(f"지원하지 않는 dataset입니다: {args.dataset}")

    query_file = temp_dir / "dataset_query.txt"
    query_file.write_text(query_text.strip() + "\n", encoding="utf-8")
    return VisualizationTarget(
        video_path=Path(video_path).resolve(),
        prompt=prompt,
        query_text=query_text,
        sample_label=sample_label,
        query_file=query_file,
    )


def _load_direct_target(
    *,
    config: DictConfig,
    args: argparse.Namespace,
    temp_dir: Path,
) -> VisualizationTarget:
    if args.video_path is not None:
        config.invoke.video_path = str(_to_abs_path(args.video_path))
    if args.query_file is not None:
        config.invoke.query_file = str(_to_abs_path(args.query_file))
    query_text = args.query_text
    if query_text is None and args.query_file is None:
        query_text = DEFAULT_DIRECT_QUERY_TEXT
    if query_text is not None:
        query_file = temp_dir / "direct_query.txt"
        query_file.write_text(query_text.strip() + "\n", encoding="utf-8")
        config.invoke.query_file = str(query_file)

    video_path = _to_abs_path(config.invoke.video_path)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    prompt = load_prompt(config.invoke)
    return VisualizationTarget(
        video_path=video_path,
        prompt=prompt,
        query_text=prompt["query"],
        sample_label=f"direct:{video_path.name}",
        query_file=_to_abs_path(config.invoke.query_file),
    )


def _resolve_target(
    *,
    config: DictConfig,
    args: argparse.Namespace,
    temp_dir: Path,
) -> VisualizationTarget:
    if args.dataset == "direct":
        return _load_direct_target(config=config, args=args, temp_dir=temp_dir)
    return _load_dataset_target(config=config, args=args, temp_dir=temp_dir)


def _configure_query_file(config: DictConfig, query_file: Path | None) -> None:
    if query_file is None:
        return
    config.invoke.query_file = str(query_file)
    for key in ("patch_selector", "frame_selector"):
        selector = config.vlm.get(key)
        if selector is not None and "query_file" in selector:
            selector.query_file = str(query_file)


def _call_with_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(callable_obj)
    accepts_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_keyword:
        return callable_obj(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(**filtered)


def _normalize_frame_selection_output(
    selection_output: torch.Tensor | FrameSelectionResult | None,
    *,
    video_path: str,
) -> FrameSelectionResult:
    if selection_output is None:
        raise ValueError("Frame selector did not return any frames.")
    if isinstance(selection_output, FrameSelectionResult):
        metadata = dict(selection_output.metadata)
        metadata.setdefault("video_path", video_path)
        return FrameSelectionResult(frames=selection_output.frames, metadata=metadata)
    if torch.is_tensor(selection_output):
        return FrameSelectionResult(
            frames=selection_output,
            metadata={
                "video_path": video_path,
                "num_frames": int(selection_output.shape[0]),
            },
        )
    raise TypeError(
        "Frame selector output must be a torch.Tensor or FrameSelectionResult."
    )


def _selector_config_dict(config: DictConfig) -> dict[str, Any]:
    selector_cfg = config.vlm.get("patch_selector")
    if selector_cfg is None:
        raise ValueError("config.vlm.patch_selector가 필요합니다.")
    return dict(OmegaConf.to_container(selector_cfg, resolve=True) or {})


def _resolve_clip_only_device(selector_cfg: dict[str, Any]) -> torch.device:
    configured = selector_cfg.get("device")
    if configured:
        return torch.device(str(configured))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_clip_only_merge_size(selector_cfg: dict[str, Any]) -> int:
    configured = selector_cfg.get("spatial_merge_size")
    if configured is None:
        return 2
    merge_size = int(configured)
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")
    return merge_size


def _compute_merge_mean_scores(
    raw_score_maps: torch.Tensor,
    *,
    merge_size: int,
) -> torch.Tensor:
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")
    return F.avg_pool2d(
        raw_score_maps.unsqueeze(1),
        kernel_size=merge_size,
        stride=merge_size,
    ).squeeze(1)


def _merge_clip_score_maps(
    raw_score_maps: torch.Tensor,
    *,
    selection_mode: str,
    merge_size: int,
    window_size: int,
    window_stride: int,
) -> torch.Tensor:
    raw_h, raw_w = int(raw_score_maps.shape[-2]), int(raw_score_maps.shape[-1])
    if raw_h % merge_size != 0 or raw_w % merge_size != 0:
        raise ValueError(
            "CLIP patch grid must be divisible by spatial_merge_size: "
            f"grid=({raw_h}, {raw_w}), merge_size={merge_size}."
        )
    if selection_mode == "naive_mean":
        return _compute_merge_mean_scores(raw_score_maps, merge_size=merge_size)
    window_maps = _compute_window_score_maps(
        raw_score_maps,
        window_size=window_size,
        window_stride=window_stride,
    )
    return _compute_sliding_window_merged_scores(
        window_maps,
        raw_height=raw_h,
        raw_width=raw_w,
        merge_size=merge_size,
        window_size=window_size,
        window_stride=window_stride,
    )


def _select_clip_only_patches(
    *,
    merged_scores: torch.Tensor,
    image_frame_scores: torch.Tensor,
    selector_cfg: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    keep_ratio = float(selector_cfg.get("keep_ratio") or 0.5)
    patch_score_threshold = selector_cfg.get("patch_score_threshold")
    target_name = str(selector_cfg.get("_target_") or "")
    uses_threshold_budget = patch_score_threshold is not None or any(
        marker in target_name
        for marker in ("selection_v3", "selection_v4", "selection_v5")
    )

    if not uses_threshold_budget:
        selected_indices, frame_metadata = _select_topk_per_frame(
            merged_scores,
            keep_ratio=keep_ratio,
        )
        return selected_indices, frame_metadata, {
            "keep_ratio": keep_ratio,
            "allocation_strategy": "topk_per_frame",
        }

    threshold = 0.0 if patch_score_threshold is None else float(patch_score_threshold)
    total_budget_value = _resolve_total_budget(
        merged_scores=merged_scores,
        keep_ratio=keep_ratio,
        total_budget=selector_cfg.get("total_budget"),
    )
    eligible_mask, eligible_counts = _compute_eligible_patch_mask(
        merged_scores,
        patch_score_threshold=threshold,
    )
    allocated_budget, raw_budget, frame_weights = _allocate_budget_with_softmax_capacities(
        image_frame_scores,
        total_budget=total_budget_value,
        capacities=eligible_counts,
        temperature=float(selector_cfg.get("temperature") or 1.0),
    )
    selected_indices, frame_metadata = _select_topk_from_eligible_patches(
        merged_scores,
        eligible_mask=eligible_mask,
        frame_scores=image_frame_scores,
        frame_weights=frame_weights,
        raw_budget=raw_budget,
        allocated_budget=allocated_budget,
        eligible_counts=eligible_counts,
    )
    selected_token_count = int(selected_indices.numel())
    return selected_indices, frame_metadata, {
        "keep_ratio": keep_ratio,
        "total_budget": int(total_budget_value),
        "allocatable_budget": int(min(total_budget_value, int(eligible_counts.sum().item()))),
        "temperature": float(selector_cfg.get("temperature") or 1.0),
        "patch_score_threshold": threshold,
        "eligible_token_count": int(eligible_counts.sum().item()),
        "underfilled_token_count": int(total_budget_value - selected_token_count),
        "allocation_strategy": "clip_only_capacity_aware",
        "frame_importance_scores": [float(score) for score in image_frame_scores.tolist()],
        "frame_softmax_weights": [float(weight) for weight in frame_weights.tolist()],
        "frame_raw_budgets": [float(budget) for budget in raw_budget.tolist()],
        "initial_frame_allocated_budgets": [int(budget) for budget in allocated_budget.tolist()],
        "final_frame_allocated_budgets": [int(budget) for budget in allocated_budget.tolist()],
        "frame_allocated_budgets": [int(budget) for budget in allocated_budget.tolist()],
    }


def _compute_clip_only_query_diagnostics(
    *,
    frame_selection: FrameSelectionResult,
    image_processor: Any,
    vision_model: Any,
    text_embeddings: torch.Tensor,
    selector_cfg: dict[str, Any],
    device: torch.device,
    clip_dtype: torch.dtype | None,
    merge_size: int,
    selection_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scoring_frames, duplication_info = _resolve_patch_scoring_frames(frame_selection)
    query_maps, _ = _compute_dense_query_score_maps(
        _prepare_frame_arrays(scoring_frames),
        image_processor=image_processor,
        vision_model=vision_model,
        text_embeddings_t=text_embeddings.transpose(0, 1).contiguous(),
        batch_size=int(selector_cfg.get("batch_size") or 1),
        device=device,
        clip_dtype=clip_dtype,
    )
    query_maps = _expand_scores_for_frame_duplication(query_maps, duplication_info)
    raw_query_scores = query_maps.detach().cpu()
    raw_query_winners = raw_query_scores.argmax(dim=1)
    merged_query_scores = _merge_query_score_maps(
        query_maps,
        selection_mode=selection_mode,
        merge_size=merge_size,
        window_size=int(selector_cfg.get("window_size") or 2),
        window_stride=int(selector_cfg.get("window_stride") or 1),
    )
    return (
        merged_query_scores,
        merged_query_scores.argmax(dim=1),
        raw_query_scores,
        raw_query_winners,
    )


def _run_clip_only_selection_diagnostics(
    *,
    config: DictConfig,
    args: argparse.Namespace,
    target: VisualizationTarget,
) -> SelectionDiagnostics:
    diagnostics_start = time.perf_counter()
    frame_selector_cfg = config.vlm.get("frame_selector") or config.get("frame_selection")
    if frame_selector_cfg is None:
        raise ValueError("config에 frame_selector/frame_selection이 없습니다.")
    _progress(args, f"sampling frames from {target.video_path.name}")
    frame_selector = instantiate(frame_selector_cfg)
    frame_selection = _normalize_frame_selection_output(
        _call_with_supported_kwargs(
            frame_selector,
            {"video_path": str(target.video_path)},
        ),
        video_path=str(target.video_path),
    )
    if frame_selection.frames is None:
        raise ValueError("Frame selector did not provide video frames.")
    _progress(
        args,
        f"sampled {int(frame_selection.frames.shape[0])} frame(s) "
        f"in {time.perf_counter() - diagnostics_start:.1f}s",
    )

    selector_cfg = _selector_config_dict(config)
    query_file = selector_cfg.get("query_file")
    if not query_file:
        raise ValueError("patch_selector.query_file이 필요합니다.")
    queries = _load_queries(query_file)
    if not queries:
        raise ValueError("Query file does not contain any queries.")

    device = _resolve_clip_only_device(selector_cfg)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(selector_cfg.get("clip_dtype"))
    clip_model_name = str(selector_cfg.get("clip_model_name") or "openai/clip-vit-large-patch14")
    clip_do_center_crop = selector_cfg.get("clip_do_center_crop")
    _progress(
        args,
        f"loading CLIP components: model={clip_model_name}, "
        f"device={device}, dtype={clip_dtype_key}",
    )
    stage_start = time.perf_counter()
    image_processor, _, vision_model, _ = _load_maskclip_components_v2(
        clip_model_name,
        _resolve_device_key(device),
        clip_dtype_key,
        clip_do_center_crop,
    )
    _progress(args, f"loaded CLIP components in {time.perf_counter() - stage_start:.1f}s")
    _progress(args, f"encoding {len(queries)} query text(s)")
    stage_start = time.perf_counter()
    clean_text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        _resolve_device_key(device),
        clip_dtype_key,
        clip_do_center_crop,
        queries,
    )
    _progress(args, f"encoded query text(s) in {time.perf_counter() - stage_start:.1f}s")
    patch_text_embeddings = _add_gaussian_noise_to_query_embeddings(
        clean_text_embeddings,
        noise_scale=float(selector_cfg.get("query_noise_scale") or 0.0),
        noise_seed=selector_cfg.get("query_noise_seed"),
    )

    scoring_frames, duplication_info = _resolve_patch_scoring_frames(frame_selection)
    _progress(
        args,
        f"computing dense patch scores for {int(scoring_frames.shape[0])} frame(s) "
        f"with batch_size={int(selector_cfg.get('batch_size') or 1)}",
    )
    stage_start = time.perf_counter()
    dense_score_maps, image_frame_scores, clip_grid = (
        _compute_dense_patch_score_maps_and_clean_frame_scores(
            _prepare_frame_arrays(scoring_frames),
            image_processor=image_processor,
            vision_model=vision_model,
            patch_text_embeddings_t=patch_text_embeddings.transpose(0, 1).contiguous(),
            frame_text_embeddings_t=clean_text_embeddings.transpose(0, 1).contiguous(),
            aggregation=str(selector_cfg.get("aggregation") or "max"),
            batch_size=int(selector_cfg.get("batch_size") or 1),
            device=device,
            clip_dtype=resolved_clip_dtype,
        )
    )
    _progress(
        args,
        f"computed dense patch scores grid={tuple(int(v) for v in dense_score_maps.shape)} "
        f"in {time.perf_counter() - stage_start:.1f}s",
    )
    dense_score_maps = _expand_scores_for_frame_duplication(
        dense_score_maps,
        duplication_info,
    )
    image_frame_scores = _expand_scores_for_frame_duplication(
        image_frame_scores,
        duplication_info,
    )

    selection_mode = _resolve_selection_mode(str(selector_cfg.get("selection_mode") or "naive_mean"))
    merge_size = _resolve_clip_only_merge_size(selector_cfg)
    merged_scores = _merge_clip_score_maps(
        dense_score_maps,
        selection_mode=selection_mode,
        merge_size=merge_size,
        window_size=int(selector_cfg.get("window_size") or 2),
        window_stride=int(selector_cfg.get("window_stride") or 1),
    )
    selected_indices, frame_metadata, selection_metadata = _select_clip_only_patches(
        merged_scores=merged_scores,
        image_frame_scores=image_frame_scores.flatten().to(dtype=torch.float32),
        selector_cfg=selector_cfg,
    )
    raw_clip_selected_indices, raw_clip_frame_metadata, raw_clip_selection_metadata = (
        _select_clip_only_patches(
            merged_scores=dense_score_maps,
            image_frame_scores=image_frame_scores.flatten().to(dtype=torch.float32),
            selector_cfg=selector_cfg,
        )
    )
    query_scores = None
    query_winners = None
    raw_clip_query_scores = None
    raw_clip_query_winners = None
    if not args.skip_query_diagnostics:
        _progress(
            args,
            "computing query diagnostics for hover labels "
            "(use --skip-query-diagnostics to skip this extra CLIP pass)",
        )
        stage_start = time.perf_counter()
        (
            query_scores,
            query_winners,
            raw_clip_query_scores,
            raw_clip_query_winners,
        ) = _compute_clip_only_query_diagnostics(
            frame_selection=frame_selection,
            image_processor=image_processor,
            vision_model=vision_model,
            text_embeddings=patch_text_embeddings,
            selector_cfg=selector_cfg,
            device=device,
            clip_dtype=resolved_clip_dtype,
            merge_size=merge_size,
            selection_mode=selection_mode,
        )
        _progress(args, f"computed query diagnostics in {time.perf_counter() - stage_start:.1f}s")
    else:
        _progress(args, "skipped query diagnostics")

    grid_t = int(merged_scores.shape[0])
    merged_h = int(merged_scores.shape[1])
    merged_w = int(merged_scores.shape[2])
    raw_h = int(dense_score_maps.shape[1])
    raw_w = int(dense_score_maps.shape[2])
    selector_metadata = {
        "selector_type": f"{Path(str(selector_cfg.get('_target_', 'clip_only'))).name}:clip_only",
        "clip_only": True,
        "selection_mode": selection_mode,
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "clip_do_center_crop": (
            None if clip_do_center_crop is None else bool(clip_do_center_crop)
        ),
        "query_file": str(Path(str(query_file)).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "query_noise_scale": float(selector_cfg.get("query_noise_scale") or 0.0),
        "query_noise_seed": selector_cfg.get("query_noise_seed"),
        "aggregation": str(selector_cfg.get("aggregation") or "max"),
        "spatial_merge_size": merge_size,
        "temporal_patch_size": 1,
        "frame_count": int(frame_selection.frames.shape[0]),
        "temporal_grid_t": grid_t,
        "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        "raw_grid_thw": [grid_t, int(clip_grid[0]), int(clip_grid[1])],
        "merged_grid_thw": [grid_t, merged_h, merged_w],
        "selected_token_count": int(selected_indices.numel()),
        "video_token_count": int(merged_scores.numel()),
        "per_frame": frame_metadata,
        **selection_metadata,
    }
    selected_mask = _build_selected_mask(selected_indices, selector_metadata)
    raw_clip_metadata = {
        **raw_clip_selection_metadata,
        "selector_type": "clip_single_patch_similarity_selection",
        "score_reduction": "none_single_raw_clip_patch",
        "spatial_merge_size": 1,
        "raw_grid_thw": [grid_t, raw_h, raw_w],
        "merged_grid_thw": [grid_t, raw_h, raw_w],
        "selected_token_count": int(raw_clip_selected_indices.numel()),
        "video_token_count": int(dense_score_maps.numel()),
        "per_frame": raw_clip_frame_metadata,
    }
    raw_clip_selected_mask = _build_selected_mask(
        raw_clip_selected_indices,
        raw_clip_metadata,
    )
    _progress(args, f"selection diagnostics finished in {time.perf_counter() - diagnostics_start:.1f}s")
    return SelectionDiagnostics(
        frame_selection=frame_selection,
        selected_indices=selected_indices.detach().cpu(),
        selector_metadata=selector_metadata,
        extraction_metadata={
            "clip_only": True,
            "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        },
        query_scores=query_scores,
        query_winners=query_winners,
        selected_mask=selected_mask,
        raw_clip_selected_mask=raw_clip_selected_mask,
        raw_clip_query_scores=raw_clip_query_scores,
        raw_clip_query_winners=raw_clip_query_winners,
        raw_clip_metadata=raw_clip_metadata,
    )


def _build_selected_mask(
    selected_indices: torch.Tensor,
    selector_metadata: dict[str, Any],
) -> torch.Tensor:
    grid_t, merged_h, merged_w = [int(value) for value in selector_metadata["merged_grid_thw"]]
    tokens_per_frame = merged_h * merged_w
    mask = torch.zeros((grid_t, merged_h, merged_w), dtype=torch.bool)
    for token_index in selected_indices.detach().cpu().flatten().tolist():
        token_index = int(token_index)
        frame_idx = token_index // tokens_per_frame
        local_index = token_index % tokens_per_frame
        if 0 <= frame_idx < grid_t:
            mask[frame_idx, local_index // merged_w, local_index % merged_w] = True
    return mask


def _expand_selected_mask_to_raw_grid(
    selected_mask: torch.Tensor,
    *,
    raw_height: int,
    raw_width: int,
) -> torch.Tensor:
    _, merged_h, merged_w = selected_mask.shape
    if raw_height % merged_h != 0 or raw_width % merged_w != 0:
        raise ValueError(
            "Raw CLIP grid must be divisible by the final selection grid: "
            f"raw=({raw_height}, {raw_width}), selected=({merged_h}, {merged_w})."
        )
    row_scale = raw_height // merged_h
    col_scale = raw_width // merged_w
    return torch.repeat_interleave(
        torch.repeat_interleave(selected_mask, repeats=row_scale, dim=1),
        repeats=col_scale,
        dim=2,
    )


def _scale_sequence_values(values: Any, multiplier: int, cast_type: type) -> list[Any] | None:
    if not isinstance(values, list):
        return None
    return [cast_type(value * multiplier) for value in values]


def _raw_grid_metadata_from_final_selection(
    selector_metadata: dict[str, Any],
    *,
    raw_selected_mask: torch.Tensor,
    raw_height: int,
    raw_width: int,
    raw_token_count: int,
) -> dict[str, Any]:
    grid_t, merged_h, merged_w = [int(value) for value in selector_metadata["merged_grid_thw"]]
    if raw_height % merged_h != 0 or raw_width % merged_w != 0:
        raise ValueError(
            "Raw CLIP grid must be divisible by the final selection grid: "
            f"raw=({raw_height}, {raw_width}), selected=({merged_h}, {merged_w})."
        )
    raw_multiplier = (raw_height // merged_h) * (raw_width // merged_w)
    raw_per_frame_counts = raw_selected_mask.reshape(grid_t, -1).sum(dim=1).tolist()
    per_frame = []
    for item, raw_count in zip(selector_metadata.get("per_frame", []), raw_per_frame_counts):
        scaled_item = dict(item)
        scaled_item["tokens_per_frame"] = raw_height * raw_width
        scaled_item["final_keep_count"] = int(raw_count)
        scaled_item["initial_keep_count"] = int(raw_count)
        scaled_item["initial_allocated_budget"] = int(raw_count)
        scaled_item["eligible_count"] = int(item.get("eligible_count", 0) * raw_multiplier)
        scaled_item["raw_budget"] = float(item.get("raw_budget", 0.0) * raw_multiplier)
        per_frame.append(scaled_item)

    raw_metadata = dict(selector_metadata)
    raw_metadata.update(
        {
            "selector_type": "final_selection_on_raw_clip_grid",
            "score_reduction": "final_selection_expanded_to_raw_clip_grid",
            "spatial_merge_size": 1,
            "raw_grid_thw": [grid_t, raw_height, raw_width],
            "merged_grid_thw": [grid_t, raw_height, raw_width],
            "selected_token_count": int(raw_selected_mask.sum().item()),
            "video_token_count": int(raw_token_count),
            "per_frame": per_frame,
        }
    )
    for key in (
        "frame_allocated_budgets",
        "final_frame_allocated_budgets",
        "initial_frame_allocated_budgets",
    ):
        scaled = _scale_sequence_values(raw_metadata.get(key), raw_multiplier, int)
        if scaled is not None:
            raw_metadata[key] = scaled
    scaled_raw_budgets = _scale_sequence_values(
        raw_metadata.get("frame_raw_budgets"),
        raw_multiplier,
        float,
    )
    if scaled_raw_budgets is not None:
        raw_metadata["frame_raw_budgets"] = scaled_raw_budgets
    return raw_metadata


def _merge_query_score_maps(
    raw_query_maps: torch.Tensor,
    *,
    selection_mode: str,
    merge_size: int,
    window_size: int,
    window_stride: int,
) -> torch.Tensor:
    frame_count, query_count, raw_h, raw_w = raw_query_maps.shape
    flat = raw_query_maps.reshape(frame_count * query_count, raw_h, raw_w)
    if selection_mode == "naive_mean":
        merged = _compute_merge_mean_scores(flat, merge_size=merge_size)
    else:
        window_maps = _compute_window_score_maps(
            flat,
            window_size=window_size,
            window_stride=window_stride,
        )
        merged = _compute_sliding_window_merged_scores(
            window_maps,
            raw_height=raw_h,
            raw_width=raw_w,
            merge_size=merge_size,
            window_size=window_size,
            window_stride=window_stride,
        )
    return merged.reshape(frame_count, query_count, merged.shape[-2], merged.shape[-1]).detach().cpu()


def _compute_dense_query_score_maps(
    frame_arrays: list[Any],
    *,
    image_processor: Any,
    vision_model: Any,
    text_embeddings_t: torch.Tensor,
    batch_size: int,
    device: torch.device,
    clip_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    chunks: list[torch.Tensor] = []
    clip_grid: tuple[int, int] | None = None
    patch_size = int(vision_model.config.patch_size)
    with torch.inference_mode():
        for start in range(0, len(frame_arrays), batch_size):
            batch_frames = frame_arrays[start : start + batch_size]
            pixel_values = image_processor(images=batch_frames, return_tensors="pt")["pixel_values"]
            if clip_dtype is None:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            else:
                pixel_values = pixel_values.to(device=device, dtype=clip_dtype, non_blocking=True)

            patch_embeddings, _ = vision_model(pixel_values)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            patch_scores = patch_embeddings @ text_embeddings_t
            clip_h = int(pixel_values.shape[-2] // patch_size)
            clip_w = int(pixel_values.shape[-1] // patch_size)
            if clip_h * clip_w != int(patch_scores.shape[1]):
                raise ValueError("CLIP patch grid inference failed during query diagnostics.")
            if clip_grid is None:
                clip_grid = (clip_h, clip_w)
            elif clip_grid != (clip_h, clip_w):
                raise ValueError(f"Inconsistent CLIP grids: {clip_grid} vs {(clip_h, clip_w)}")
            chunks.append(patch_scores.view(patch_scores.shape[0], clip_h, clip_w, patch_scores.shape[-1]))

    if clip_grid is None:
        raise ValueError("No frames were provided for query diagnostics.")
    return torch.cat(chunks, dim=0).permute(0, 3, 1, 2).contiguous(), clip_grid


def _frames_for_temporal_grid(
    frame_selection: FrameSelectionResult,
    selector_metadata: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    frames = frame_selection.frames.detach().cpu()
    grid_t = int(selector_metadata["merged_grid_thw"][0])
    temporal_patch_size = int(selector_metadata.get("temporal_patch_size") or 1)
    if int(frames.shape[0]) == grid_t:
        display_frames = frames
    elif int(frames.shape[0]) == grid_t * temporal_patch_size:
        display_frames = frames[::temporal_patch_size]
    else:
        indices = torch.linspace(0, int(frames.shape[0]) - 1, grid_t).round().long()
        display_frames = frames[indices]

    metadata = frame_selection.metadata
    sampled = metadata.get("sampled_indices_before_duplication") or metadata.get("sampled_indices")
    if isinstance(sampled, list) and len(sampled) == int(frames.shape[0]) and len(sampled) != grid_t:
        sampled = sampled[:: max(1, temporal_patch_size)]
    if not isinstance(sampled, list) or len(sampled) < grid_t:
        sampled = list(range(grid_t))
    return display_frames[:grid_t], [int(value) for value in sampled[:grid_t]]


def _frame_to_uint8(frame: torch.Tensor) -> Any:
    import numpy as np

    array = frame.detach().cpu().numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _display_frame_indices(frame_count: int, max_panels: int) -> list[int]:
    if max_panels <= 0 or frame_count <= max_panels:
        return list(range(frame_count))
    return sorted(set(torch.linspace(0, frame_count - 1, max_panels).round().long().tolist()))


def _query_label(queries: list[str], query_index: int | None) -> str:
    if query_index is None or query_index < 0 or query_index >= len(queries):
        return "n/a"
    return f"q{query_index}: {queries[query_index]}"


def _patch_hover_rows(
    *,
    frame_idx: int,
    selected_mask: torch.Tensor,
    query_scores: torch.Tensor | None,
    query_winners: torch.Tensor | None,
    queries: list[str],
) -> tuple[list[float], list[float], list[list[Any]], list[bool]]:
    _, merged_h, merged_w = selected_mask.shape
    xs: list[float] = []
    ys: list[float] = []
    customdata: list[list[Any]] = []
    selected_flags: list[bool] = []
    for row in range(merged_h):
        for col in range(merged_w):
            selected = bool(selected_mask[frame_idx, row, col].item())
            winner_index = None
            score = None
            if query_winners is not None:
                winner_index = int(query_winners[frame_idx, row, col].item())
            if query_scores is not None and winner_index is not None:
                score = float(query_scores[frame_idx, winner_index, row, col].item())
            xs.append(col + 0.5)
            ys.append(row + 0.5)
            customdata.append(
                [
                    frame_idx,
                    row,
                    col,
                    "selected" if selected else "not selected",
                    "" if score is None else f"{score:.4f}",
                    _query_label(queries, winner_index),
                ]
            )
            selected_flags.append(selected)
    return xs, ys, customdata, selected_flags


def _raw_clip_hover_rows(
    *,
    frame_idx: int,
    raw_mask: torch.Tensor,
    raw_query_scores: torch.Tensor | None,
    raw_query_winners: torch.Tensor | None,
    queries: list[str],
) -> tuple[list[float], list[float], list[list[Any]], list[bool]]:
    _, raw_h, raw_w = raw_mask.shape
    xs: list[float] = []
    ys: list[float] = []
    customdata: list[list[Any]] = []
    selected_flags: list[bool] = []
    for row in range(raw_h):
        for col in range(raw_w):
            selected = bool(raw_mask[frame_idx, row, col].item())
            winner_index = None
            score = None
            if raw_query_winners is not None:
                winner_index = int(raw_query_winners[frame_idx, row, col].item())
            if raw_query_scores is not None and winner_index is not None:
                score = float(raw_query_scores[frame_idx, winner_index, row, col].item())
            xs.append(col + 0.5)
            ys.append(row + 0.5)
            customdata.append(
                [
                    frame_idx,
                    row,
                    col,
                    "selected" if selected else "not selected",
                    "" if score is None else f"{score:.4f}",
                    _query_label(queries, winner_index),
                ]
            )
            selected_flags.append(selected)
    return xs, ys, customdata, selected_flags


def _subplot_axis_refs(figure: Any, row: int, col: int) -> tuple[str, str]:
    grid_ref = getattr(figure, "_grid_ref", None)
    if grid_ref is None:
        raise ValueError("Plotly subplot grid metadata is unavailable.")
    subplot_ref = grid_ref[row - 1][col - 1][0]
    trace_kwargs = subplot_ref.trace_kwargs
    return str(trace_kwargs["xaxis"]), str(trace_kwargs["yaxis"])


def _patch_shapes(
    *,
    xref: str,
    yref: str,
    row: int,
    col: int,
    frame_idx: int,
    selected_mask: torch.Tensor,
    image_height: int,
    image_width: int,
    selected_fill: str = "rgba(16, 185, 129, 0.32)",
    selected_line: str = "rgba(5, 150, 105, 0.96)",
    unselected_fill: str = "rgba(100, 116, 139, 0.055)",
    unselected_line: str = "rgba(71, 85, 105, 0.17)",
) -> list[dict[str, Any]]:
    del row, col
    _, merged_h, merged_w = selected_mask.shape
    patch_w = image_width / merged_w
    patch_h = image_height / merged_h
    shapes: list[dict[str, Any]] = []
    for patch_row in range(merged_h):
        for patch_col in range(merged_w):
            selected = bool(selected_mask[frame_idx, patch_row, patch_col].item())
            fill = selected_fill if selected else unselected_fill
            line = selected_line if selected else unselected_line
            width = 1.4 if selected else 0.35
            shapes.append(
                {
                    "type": "rect",
                    "xref": xref,
                    "yref": yref,
                    "x0": patch_col * patch_w,
                    "x1": (patch_col + 1) * patch_w,
                    "y0": patch_row * patch_h,
                    "y1": (patch_row + 1) * patch_h,
                    "fillcolor": fill,
                    "line": {"color": line, "width": width},
                }
            )
    return shapes


def _raw_clip_patch_shapes(
    *,
    xref: str,
    yref: str,
    row: int,
    col: int,
    frame_idx: int,
    raw_mask: torch.Tensor,
    image_height: int,
    image_width: int,
) -> list[dict[str, Any]]:
    del row, col
    _, raw_h, raw_w = raw_mask.shape
    patch_w = image_width / raw_w
    patch_h = image_height / raw_h
    shapes: list[dict[str, Any]] = []
    for patch_row in range(raw_h):
        for patch_col in range(raw_w):
            if not bool(raw_mask[frame_idx, patch_row, patch_col].item()):
                continue
            shapes.append(
                {
                    "type": "rect",
                    "xref": xref,
                    "yref": yref,
                    "x0": patch_col * patch_w,
                    "x1": (patch_col + 1) * patch_w,
                    "y0": patch_row * patch_h,
                    "y1": (patch_row + 1) * patch_h,
                    "fillcolor": "rgba(245, 158, 11, 0.14)",
                    "line": {"color": "rgba(217, 119, 6, 0.96)", "width": 1.1},
                }
            )
    return shapes


def _build_figure(
    *,
    go: Any,
    make_subplots: Any,
    diagnostics: SelectionDiagnostics,
    target: VisualizationTarget,
    max_frame_panels: int,
    panel_cols: int,
) -> Any:
    metadata = diagnostics.selector_metadata
    qwen_mask = diagnostics.selected_mask
    if qwen_mask is None:
        raise ValueError("Selected patch mask is missing.")
    clip_mask = diagnostics.raw_clip_selected_mask
    if clip_mask is None:
        clip_mask = qwen_mask
    clip_metadata = diagnostics.raw_clip_metadata or metadata

    display_frames, sampled_indices = _frames_for_temporal_grid(
        diagnostics.frame_selection,
        metadata,
    )
    grid_t, qwen_h, qwen_w = qwen_mask.shape
    qwen_tokens_per_frame = qwen_h * qwen_w
    qwen_counts = qwen_mask.reshape(grid_t, -1).sum(dim=1).tolist()
    qwen_unselected_counts = [qwen_tokens_per_frame - int(value) for value in qwen_counts]
    clip_tokens_per_frame = int(clip_mask.shape[1] * clip_mask.shape[2])
    clip_counts = clip_mask.reshape(clip_mask.shape[0], -1).sum(dim=1).tolist()
    frame_scores = metadata.get("frame_importance_scores") or [
        item.get("frame_importance_score") for item in metadata.get("per_frame", [])
    ]
    frame_scores = [float(value) for value in frame_scores] if frame_scores else None
    frame_budgets = metadata.get("frame_allocated_budgets") or metadata.get("final_frame_allocated_budgets")
    frame_budgets = [int(value) for value in frame_budgets] if frame_budgets else None
    queries = [str(query) for query in metadata.get("queries", [])]

    shown_frames = _display_frame_indices(grid_t, max_frame_panels)
    panel_cols = max(1, min(int(panel_cols), max(1, len(shown_frames))))
    panel_rows = int(math.ceil(len(shown_frames) / panel_cols))
    section_count = 2
    total_rows = 1 + panel_rows * section_count
    specs = [[{"colspan": panel_cols, "secondary_y": True}, *([None] * (panel_cols - 1))]]
    for _ in range(panel_rows * section_count):
        specs.append([{} for _ in range(panel_cols)])

    subplot_titles = ["Temporal Patch Allocation"]
    for section_title, mask, tokens_per_frame in (
        ("CLIP raw patch selection", clip_mask, clip_tokens_per_frame),
        ("Qwen merged final input", qwen_mask, qwen_tokens_per_frame),
    ):
        counts = mask.reshape(mask.shape[0], -1).sum(dim=1).tolist()
        for frame_idx in shown_frames:
            original_index = sampled_indices[frame_idx] if frame_idx < len(sampled_indices) else frame_idx
            subplot_titles.append(
                f"{section_title}<br>t={frame_idx} frame={original_index} selected={int(counts[frame_idx])}/{tokens_per_frame}"
            )

    chart_row_height = 0.22
    panel_area_height = 1.0 - chart_row_height
    figure = make_subplots(
        rows=total_rows,
        cols=panel_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.055,
        horizontal_spacing=0.035,
        row_heights=[chart_row_height] + [panel_area_height / max(panel_rows * section_count, 1)] * (panel_rows * section_count),
    )

    temporal_x = list(range(grid_t))
    figure.add_trace(
        go.Bar(
            x=temporal_x,
            y=qwen_unselected_counts,
            name="Qwen not selected",
            marker={"color": "rgba(148, 163, 184, 0.52)"},
            hovertemplate="t=%{x}<br>not_selected=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=temporal_x,
            y=[int(value) for value in qwen_counts],
            name="Qwen selected",
            marker={"color": "rgba(16, 185, 129, 0.82)"},
            hovertemplate="t=%{x}<br>selected=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=temporal_x,
            y=[int(value) for value in clip_counts],
            mode="lines+markers",
            name="CLIP raw selected",
            line={"color": "#2563EB", "width": 1.8, "dash": "dash"},
            marker={"size": 5},
            hovertemplate=f"t=%{{x}}<br>clip_raw_selected=%{{y}}/{clip_tokens_per_frame}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    if frame_scores is not None and len(frame_scores) == grid_t:
        figure.add_trace(
            go.Scatter(
                x=temporal_x,
                y=frame_scores,
                mode="lines+markers",
                name="Frame query score",
                line={"color": "#2563EB", "width": 2.2},
                marker={"size": 6},
                hovertemplate="t=%{x}<br>frame_score=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    if frame_budgets is not None and len(frame_budgets) == grid_t:
        figure.add_trace(
            go.Scatter(
                x=temporal_x,
                y=frame_budgets,
                mode="lines+markers",
                name="Allocated budget",
                line={"color": "#F59E0B", "width": 1.8, "dash": "dot"},
                marker={"size": 5},
                hovertemplate="t=%{x}<br>budget=%{y}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    overlay_shapes: list[dict[str, Any]] = []
    sections = [
        {
            "mask": clip_mask,
            "query_scores": diagnostics.raw_clip_query_scores,
            "query_winners": diagnostics.raw_clip_query_winners,
            "selected_fill": "rgba(245, 158, 11, 0.28)",
            "selected_line": "rgba(217, 119, 6, 0.96)",
            "hover_name": "CLIP raw patch hover",
        },
        {
            "mask": qwen_mask,
            "query_scores": diagnostics.query_scores,
            "query_winners": diagnostics.query_winners,
            "selected_fill": "rgba(16, 185, 129, 0.32)",
            "selected_line": "rgba(5, 150, 105, 0.96)",
            "hover_name": "Qwen merged patch hover",
        },
    ]
    for section_number, section in enumerate(sections):
        section_mask = section["mask"]
        section_h, section_w = int(section_mask.shape[1]), int(section_mask.shape[2])
        for panel_number, frame_idx in enumerate(shown_frames):
            subplot_row = 2 + section_number * panel_rows + panel_number // panel_cols
            subplot_col = 1 + panel_number % panel_cols
            frame = _frame_to_uint8(display_frames[frame_idx])
            image_height, image_width = frame.shape[:2]
            figure.add_trace(go.Image(z=frame, hoverinfo="skip"), row=subplot_row, col=subplot_col)
            xref, yref = _subplot_axis_refs(figure, subplot_row, subplot_col)
            overlay_shapes.extend(
                _patch_shapes(
                    xref=xref,
                    yref=yref,
                    row=subplot_row,
                    col=subplot_col,
                    frame_idx=frame_idx,
                    selected_mask=section_mask,
                    image_height=image_height,
                    image_width=image_width,
                    selected_fill=str(section["selected_fill"]),
                    selected_line=str(section["selected_line"]),
                )
            )
            xs, ys, customdata, selected_flags = _patch_hover_rows(
                frame_idx=frame_idx,
                selected_mask=section_mask,
                query_scores=section["query_scores"],
                query_winners=section["query_winners"],
                queries=queries,
            )
            patch_w = image_width / section_w
            patch_h = image_height / section_h
            figure.add_trace(
                go.Scatter(
                    x=[value * patch_w for value in xs],
                    y=[value * patch_h for value in ys],
                    mode="markers",
                    name=str(section["hover_name"]),
                    showlegend=False,
                    marker={
                        "size": 8,
                        "color": [
                            "rgba(16, 185, 129, 0.02)"
                            if flag
                            else "rgba(148, 163, 184, 0.01)"
                            for flag in selected_flags
                        ],
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "t=%{customdata[0]}<br>"
                        "patch(row,col)=(%{customdata[1]}, %{customdata[2]})<br>"
                        "state=%{customdata[3]}<br>"
                        "query_score=%{customdata[4]}<br>"
                        "top_query=%{customdata[5]}<extra></extra>"
                    ),
                ),
                row=subplot_row,
                col=subplot_col,
            )
            figure.update_xaxes(
                visible=False,
                range=[0, image_width],
                constrain="domain",
                row=subplot_row,
                col=subplot_col,
            )
            figure.update_yaxes(
                visible=False,
                range=[image_height, 0],
                scaleanchor=xref,
                scaleratio=1,
                constrain="domain",
                row=subplot_row,
                col=subplot_col,
            )

    figure.update_layout(shapes=overlay_shapes)

    query_preview = "; ".join(f"q{idx}: {query}" for idx, query in enumerate(queries[:6]))
    if len(queries) > 6:
        query_preview += f"; ... (+{len(queries) - 6})"
    safe_query = html.escape(query_preview[:360])
    safe_sample = html.escape(target.sample_label)
    title_stats = (
        f"selector={metadata.get('selector_type')} | "
        f"qwen_selected={metadata.get('selected_token_count')}/{metadata.get('video_token_count')} "
        f"grid={metadata.get('merged_grid_thw')} | "
        f"clip_view_selected={clip_metadata.get('selected_token_count')}/{clip_metadata.get('video_token_count')} "
        f"grid={clip_metadata.get('merged_grid_thw')} | "
        f"aggregation={metadata.get('aggregation')} | threshold={metadata.get('patch_score_threshold', 'n/a')}"
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": f"Selected Patch Visualization - {safe_sample}<br><sup>{html.escape(title_stats)}<br>queries: {safe_query}</sup>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
        },
        barmode="stack",
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
        },
        margin={"l": 64, "r": 48, "t": 185, "b": 56},
        height=max(1120, 360 + panel_rows * section_count * 340),
        uniformtext={"mode": "hide", "minsize": 9},
    )
    figure.update_annotations(font={"size": 11}, align="center")
    figure.update_xaxes(title_text="temporal grid index", row=1, col=1)
    figure.update_yaxes(title_text="patch count", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="frame score", row=1, col=1, secondary_y=True)
    return figure


def _split_output_paths(output_path: Path) -> dict[str, Path]:
    suffix = output_path.suffix or ".html"
    return {
        "temporal": output_path.with_name(f"{output_path.stem}_temporal{suffix}"),
        "clip": output_path.with_name(f"{output_path.stem}_clip{suffix}"),
        "qwen": output_path.with_name(f"{output_path.stem}_qwen{suffix}"),
    }


def _plotly_html_config(output_path: Path, figure: Any) -> dict[str, Any]:
    width = int(figure.layout.width or 1600)
    height = int(figure.layout.height or 1000)
    return {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": output_path.with_suffix("").name,
            "width": width,
            "height": height,
            "scale": 1,
        },
        "displaylogo": False,
    }


def _title_context(
    *,
    target: VisualizationTarget,
    metadata: dict[str, Any],
    queries: list[str],
    view_stats: str,
) -> dict[str, Any]:
    query_preview = "; ".join(f"q{idx}: {query}" for idx, query in enumerate(queries[:6]))
    if len(queries) > 6:
        query_preview += f"; ... (+{len(queries) - 6})"
    title_stats = (
        f"{view_stats} | aggregation={metadata.get('aggregation')} | "
        f"threshold={metadata.get('patch_score_threshold', 'n/a')}"
    )
    return {
        "sample": html.escape(target.sample_label),
        "stats": html.escape(title_stats),
        "queries": html.escape(query_preview[:360]),
    }


def _build_temporal_figure(
    *,
    go: Any,
    make_subplots: Any,
    diagnostics: SelectionDiagnostics,
    target: VisualizationTarget,
) -> Any:
    metadata = diagnostics.selector_metadata
    qwen_mask = diagnostics.selected_mask
    if qwen_mask is None:
        raise ValueError("Selected patch mask is missing.")
    clip_mask = diagnostics.raw_clip_selected_mask
    if clip_mask is None:
        clip_mask = qwen_mask
    clip_metadata = diagnostics.raw_clip_metadata or metadata

    grid_t, qwen_h, qwen_w = qwen_mask.shape
    qwen_tokens_per_frame = qwen_h * qwen_w
    qwen_counts = qwen_mask.reshape(grid_t, -1).sum(dim=1).tolist()
    qwen_unselected_counts = [qwen_tokens_per_frame - int(value) for value in qwen_counts]
    clip_tokens_per_frame = int(clip_mask.shape[1] * clip_mask.shape[2])
    clip_counts = clip_mask.reshape(clip_mask.shape[0], -1).sum(dim=1).tolist()
    frame_scores = metadata.get("frame_importance_scores") or [
        item.get("frame_importance_score") for item in metadata.get("per_frame", [])
    ]
    frame_scores = [float(value) for value in frame_scores] if frame_scores else None
    frame_budgets = metadata.get("frame_allocated_budgets") or metadata.get("final_frame_allocated_budgets")
    frame_budgets = [int(value) for value in frame_budgets] if frame_budgets else None
    queries = [str(query) for query in metadata.get("queries", [])]

    figure = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=["Temporal Patch Allocation"],
    )
    temporal_x = list(range(grid_t))
    figure.add_trace(
        go.Bar(
            x=temporal_x,
            y=qwen_unselected_counts,
            name="Qwen not selected",
            marker={"color": "rgba(148, 163, 184, 0.52)"},
            hovertemplate="t=%{x}<br>not_selected=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=temporal_x,
            y=[int(value) for value in qwen_counts],
            name="Qwen selected",
            marker={"color": "rgba(16, 185, 129, 0.82)"},
            hovertemplate=f"t=%{{x}}<br>qwen_selected=%{{y}}/{qwen_tokens_per_frame}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=temporal_x,
            y=[int(value) for value in clip_counts],
            mode="lines+markers",
            name="CLIP raw selected",
            line={"color": "#2563EB", "width": 1.8, "dash": "dash"},
            marker={"size": 5},
            hovertemplate=f"t=%{{x}}<br>clip_raw_selected=%{{y}}/{clip_tokens_per_frame}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    if frame_scores is not None and len(frame_scores) == grid_t:
        figure.add_trace(
            go.Scatter(
                x=temporal_x,
                y=frame_scores,
                mode="lines+markers",
                name="Frame query score",
                line={"color": "#2563EB", "width": 2.2},
                marker={"size": 6},
                hovertemplate="t=%{x}<br>frame_score=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    if frame_budgets is not None and len(frame_budgets) == grid_t:
        figure.add_trace(
            go.Scatter(
                x=temporal_x,
                y=frame_budgets,
                mode="lines+markers",
                name="Allocated budget",
                line={"color": "#F59E0B", "width": 1.8, "dash": "dot"},
                marker={"size": 5},
                hovertemplate="t=%{x}<br>budget=%{y}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    title = _title_context(
        target=target,
        metadata=metadata,
        queries=queries,
        view_stats=(
            f"qwen_selected={metadata.get('selected_token_count')}/{metadata.get('video_token_count')} "
            f"grid={metadata.get('merged_grid_thw')} | "
            f"clip_view_selected={clip_metadata.get('selected_token_count')}/{clip_metadata.get('video_token_count')} "
            f"grid={clip_metadata.get('merged_grid_thw')}"
        ),
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"Temporal Patch Allocation - {title['sample']}<br>"
                f"<sup>{title['stats']}<br>queries: {title['queries']}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="stack",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "left", "x": 0.0},
        margin={"l": 64, "r": 48, "t": 145, "b": 56},
        height=560,
    )
    figure.update_annotations(font={"size": 12}, align="center")
    figure.update_xaxes(title_text="temporal grid index", row=1, col=1)
    figure.update_yaxes(title_text="patch count", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="frame score", row=1, col=1, secondary_y=True)
    return figure


def _build_patch_view_figure(
    *,
    go: Any,
    make_subplots: Any,
    diagnostics: SelectionDiagnostics,
    target: VisualizationTarget,
    mask: torch.Tensor,
    view_metadata: dict[str, Any],
    view_title: str,
    hover_name: str,
    selected_fill: str,
    selected_line: str,
    query_scores: torch.Tensor | None,
    query_winners: torch.Tensor | None,
    max_frame_panels: int,
    panel_cols: int,
) -> Any:
    metadata = diagnostics.selector_metadata
    display_frames, sampled_indices = _frames_for_temporal_grid(
        diagnostics.frame_selection,
        metadata,
    )
    grid_t, grid_h, grid_w = mask.shape
    tokens_per_frame = grid_h * grid_w
    selected_counts = mask.reshape(grid_t, -1).sum(dim=1).tolist()
    queries = [str(query) for query in metadata.get("queries", [])]
    shown_frames = _display_frame_indices(grid_t, max_frame_panels)
    panel_cols = max(1, min(int(panel_cols), max(1, len(shown_frames))))
    panel_rows = int(math.ceil(len(shown_frames) / panel_cols))
    specs = [[{} for _ in range(panel_cols)] for _ in range(panel_rows)]
    subplot_titles = []
    for frame_idx in shown_frames:
        original_index = sampled_indices[frame_idx] if frame_idx < len(sampled_indices) else frame_idx
        subplot_titles.append(
            f"t={frame_idx} frame={original_index}<br>selected {int(selected_counts[frame_idx])}/{tokens_per_frame}"
        )

    figure = make_subplots(
        rows=panel_rows,
        cols=panel_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.075,
        horizontal_spacing=0.035,
    )
    overlay_shapes: list[dict[str, Any]] = []
    for panel_number, frame_idx in enumerate(shown_frames):
        subplot_row = 1 + panel_number // panel_cols
        subplot_col = 1 + panel_number % panel_cols
        frame = _frame_to_uint8(display_frames[frame_idx])
        image_height, image_width = frame.shape[:2]
        figure.add_trace(go.Image(z=frame, hoverinfo="skip"), row=subplot_row, col=subplot_col)
        xref, yref = _subplot_axis_refs(figure, subplot_row, subplot_col)
        overlay_shapes.extend(
            _patch_shapes(
                xref=xref,
                yref=yref,
                row=subplot_row,
                col=subplot_col,
                frame_idx=frame_idx,
                selected_mask=mask,
                image_height=image_height,
                image_width=image_width,
                selected_fill=selected_fill,
                selected_line=selected_line,
            )
        )
        xs, ys, customdata, selected_flags = _patch_hover_rows(
            frame_idx=frame_idx,
            selected_mask=mask,
            query_scores=query_scores,
            query_winners=query_winners,
            queries=queries,
        )
        patch_w = image_width / grid_w
        patch_h = image_height / grid_h
        figure.add_trace(
            go.Scatter(
                x=[value * patch_w for value in xs],
                y=[value * patch_h for value in ys],
                mode="markers",
                name=hover_name,
                showlegend=False,
                marker={
                    "size": 8,
                    "color": [
                        "rgba(16, 185, 129, 0.02)" if flag else "rgba(148, 163, 184, 0.01)"
                        for flag in selected_flags
                    ],
                },
                customdata=customdata,
                hovertemplate=(
                    "t=%{customdata[0]}<br>"
                    "patch(row,col)=(%{customdata[1]}, %{customdata[2]})<br>"
                    "state=%{customdata[3]}<br>"
                    "query_score=%{customdata[4]}<br>"
                    "top_query=%{customdata[5]}<extra></extra>"
                ),
            ),
            row=subplot_row,
            col=subplot_col,
        )
        figure.update_xaxes(
            visible=False,
            range=[0, image_width],
            constrain="domain",
            row=subplot_row,
            col=subplot_col,
        )
        figure.update_yaxes(
            visible=False,
            range=[image_height, 0],
            scaleanchor=xref,
            scaleratio=1,
            constrain="domain",
            row=subplot_row,
            col=subplot_col,
        )

    title = _title_context(
        target=target,
        metadata=metadata,
        queries=queries,
        view_stats=(
            f"selector={view_metadata.get('selector_type')} | "
            f"selected={view_metadata.get('selected_token_count')}/{view_metadata.get('video_token_count')} | "
            f"grid={view_metadata.get('merged_grid_thw')}"
        ),
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"{html.escape(view_title)} - {title['sample']}<br>"
                f"<sup>{title['stats']}<br>queries: {title['queries']}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
        },
        hovermode="closest",
        margin={"l": 48, "r": 48, "t": 175, "b": 56},
        height=max(780, panel_rows * 360 + 240),
        shapes=overlay_shapes,
    )
    figure.update_annotations(font={"size": 11}, align="center")
    return figure


def _build_split_figures(
    *,
    go: Any,
    make_subplots: Any,
    diagnostics: SelectionDiagnostics,
    target: VisualizationTarget,
    max_frame_panels: int,
    panel_cols: int,
) -> dict[str, Any]:
    qwen_mask = diagnostics.selected_mask
    if qwen_mask is None:
        raise ValueError("Selected patch mask is missing.")
    clip_mask = diagnostics.raw_clip_selected_mask
    if clip_mask is None:
        clip_mask = qwen_mask
    clip_metadata = diagnostics.raw_clip_metadata or diagnostics.selector_metadata
    return {
        "temporal": _build_temporal_figure(
            go=go,
            make_subplots=make_subplots,
            diagnostics=diagnostics,
            target=target,
        ),
        "clip": _build_patch_view_figure(
            go=go,
            make_subplots=make_subplots,
            diagnostics=diagnostics,
            target=target,
            mask=clip_mask,
            view_metadata=clip_metadata,
            view_title="CLIP Raw Patch Selection",
            hover_name="CLIP raw patch hover",
            selected_fill="rgba(37, 99, 235, 0.28)",
            selected_line="rgba(29, 78, 216, 0.96)",
            query_scores=diagnostics.raw_clip_query_scores,
            query_winners=diagnostics.raw_clip_query_winners,
            max_frame_panels=max_frame_panels,
            panel_cols=panel_cols,
        ),
        "qwen": _build_patch_view_figure(
            go=go,
            make_subplots=make_subplots,
            diagnostics=diagnostics,
            target=target,
            mask=qwen_mask,
            view_metadata=diagnostics.selector_metadata,
            view_title="Qwen Merged Final Input",
            hover_name="Qwen merged patch hover",
            selected_fill="rgba(16, 185, 129, 0.32)",
            selected_line="rgba(5, 150, 105, 0.96)",
            query_scores=diagnostics.query_scores,
            query_winners=diagnostics.query_winners,
            max_frame_panels=max_frame_panels,
            panel_cols=panel_cols,
        ),
    }


def main() -> None:
    args = _parse_args()
    script_start = time.perf_counter()
    _progress(args, "loading Plotly backend")
    go, make_subplots = _load_plot_backend()
    _progress(args, f"loading config: {args.config}")
    config = OmegaConf.load(str(args.config))
    _apply_common_config_overrides(config, args)
    output_path, coerced_to_html = _resolve_output_path(args.output)

    with tempfile.TemporaryDirectory(prefix="show_selected_patch_") as temp_name:
        temp_dir = Path(temp_name)
        _progress(args, "resolving visualization target")
        target = _resolve_target(config=config, args=args, temp_dir=temp_dir)
        _configure_query_file(config, target.query_file)
        _progress(args, f"target video={target.video_path}")
        _progress(args, f"target query={target.query_text}")
        diagnostics = _run_clip_only_selection_diagnostics(
            config=config,
            args=args,
            target=target,
        )

        _progress(args, "building Plotly figures")
        stage_start = time.perf_counter()
        figures = _build_split_figures(
            go=go,
            make_subplots=make_subplots,
            diagnostics=diagnostics,
            target=target,
            max_frame_panels=args.max_frame_panels,
            panel_cols=args.panel_cols,
        )
        _progress(args, f"built Plotly figures in {time.perf_counter() - stage_start:.1f}s")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths = _split_output_paths(output_path)
        for name, figure in figures.items():
            split_output_path = output_paths[name]
            _progress(args, f"writing {name} HTML: {split_output_path}")
            figure.write_html(
                str(split_output_path),
                include_plotlyjs="cdn",
                full_html=True,
                config=_plotly_html_config(split_output_path, figure),
            )

    _progress(args, f"done in {time.perf_counter() - script_start:.1f}s")
    print("Outputs     :")
    for name, split_output_path in output_paths.items():
        print(f"  {name:<8}: {split_output_path}")
    if coerced_to_html:
        print("Note        : output suffix was changed to .html")
    print(f"Dataset     : {args.dataset}")
    print(f"Sample      : {target.sample_label}")
    print(f"Video       : {target.video_path}")
    print(f"Query       : {target.query_text}")
    print(f"Selected    : {diagnostics.selector_metadata.get('selected_token_count')}/"
          f"{diagnostics.selector_metadata.get('video_token_count')}")
    if args.show:
        for split_output_path in output_paths.values():
            webbrowser.open(split_output_path.as_uri())


if __name__ == "__main__":
    main()
