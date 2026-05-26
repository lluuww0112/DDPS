from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import tempfile
import textwrap
import time
import webbrowser
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_OUTPUT = REPO_ROOT / "test" / "artifacts" / "videomme_gumbel_frames.png"
MATPLOTLIB_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval.videomme as videomme
from model.video.selection import uniform_sampling


OPTION_LABEL_PATTERN = re.compile(
    r"^\s*(?:option\s*)?[\(\[]?[A-D][\)\]\.\:\-]\s*",
    re.IGNORECASE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly pick a Video-MME video/question pair, run Gumbel frame "
            "selection, and render the selected frames."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "gumbel.yaml",
        help="Experiment config containing frame_selection.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=CONFIG_DIR / "eval.yaml",
        help="Eval config containing the videomme section.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--video-map-file", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for random Video-MME QA sampling. Omit for non-deterministic sampling.",
    )
    parser.add_argument("--candidate-num-frames", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--gumbel-temperature", type=float, default=None)
    parser.add_argument("--gumbel-seed", type=int, default=None)
    parser.add_argument("--clip-dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--panel-cols", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true", help="Open the image after writing it.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _progress(args: argparse.Namespace, message: str) -> None:
    if not args.quiet:
        print(f"[videomme-gumbel] {message}", flush=True)


def _to_abs_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def _load_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with `pip install matplotlib`.") from exc
    return plt


def _load_runtime_config(args: argparse.Namespace) -> DictConfig:
    experiment_config = OmegaConf.load(str(_to_abs_path(args.config)))
    eval_config = OmegaConf.load(str(_to_abs_path(args.eval_config)))
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must be a DictConfig: {args.config}")
    if not isinstance(eval_config, DictConfig):
        raise TypeError(f"Eval config must be a DictConfig: {args.eval_config}")
    runtime_config = OmegaConf.merge(experiment_config, eval_config)
    if runtime_config.get("videomme") is None:
        raise ValueError("The merged config must contain a `videomme` section.")
    if runtime_config.get("frame_selection") is None:
        raise ValueError("The experiment config must contain a `frame_selection` section.")
    return runtime_config


def _apply_dataset_overrides(eval_config: DictConfig, args: argparse.Namespace) -> None:
    overrides = {
        "dataset_root": args.dataset_root,
        "annotation_file": args.annotation_file,
        "videos_dir": args.videos_dir,
        "video_map_file": args.video_map_file,
    }
    for key, value in overrides.items():
        if value is not None:
            eval_config[key] = str(_to_abs_path(value))


def _apply_selector_overrides(config: DictConfig, args: argparse.Namespace, query_file: Path) -> None:
    selector_config = config.frame_selection
    selector_config.query_file = str(query_file)

    overrides = {
        "candidate_num_frames": args.candidate_num_frames,
        "num_frames": args.num_frames,
        "gumbel_temperature": args.gumbel_temperature,
        "gumbel_seed": args.gumbel_seed,
        "clip_dtype": args.clip_dtype,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_side": args.max_side,
    }
    for key, value in overrides.items():
        if value is not None:
            selector_config[key] = value

    if config.get("invoke") is None:
        config.invoke = OmegaConf.create({})
    config.invoke.query_file = str(query_file)


def _load_samples(args: argparse.Namespace, config: DictConfig) -> list[videomme.VideoMMEQuestion]:
    eval_config = config.videomme
    _apply_dataset_overrides(eval_config, args)
    _, annotation_file, videos_dir, video_map_file = videomme._resolve_dataset_layout(eval_config)
    indexed_videos = videomme._index_videos(videos_dir)
    video_lookup = videomme._build_video_lookup(indexed_videos)
    video_map = videomme._load_video_map(video_map_file)
    samples, _ = videomme._load_samples(
        annotation_file=annotation_file,
        indexed_videos=indexed_videos,
        video_lookup=video_lookup,
        video_map=video_map,
    )

    start_index = videomme._resolve_optional_int(eval_config.get("start_index")) or 0
    limit = videomme._resolve_optional_int(eval_config.get("limit"))
    if start_index > 0:
        samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No Video-MME samples were loaded.")
    return samples


def _sample_matches_id(sample: videomme.VideoMMEQuestion, sample_id: str) -> bool:
    target = sample_id.strip().casefold()
    fields = (
        sample.video_id,
        sample.video_name,
        sample.question_id,
        f"{sample.video_id}/{sample.question_id}",
    )
    return any(str(field).strip().casefold() == target for field in fields)


def _pick_sample(
    samples: list[videomme.VideoMMEQuestion],
    args: argparse.Namespace,
) -> tuple[int, videomme.VideoMMEQuestion]:
    if args.sample_id:
        for index, sample in enumerate(samples):
            if _sample_matches_id(sample, args.sample_id):
                return index, sample
        raise ValueError(f"No sample matched --sample-id={args.sample_id!r}.")

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise IndexError(f"--sample-index out of range: {args.sample_index}/{len(samples)}")
        return args.sample_index, samples[args.sample_index]

    rng = random.Random(args.sample_seed)
    index = rng.randrange(len(samples))
    return index, samples[index]


def _frame_to_uint8(frame: torch.Tensor) -> Any:
    import numpy as np

    array = frame.detach().cpu().numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": [int(dim) for dim in value.shape],
            "tensor_dtype": str(value.dtype),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_frame_selection_query_text(sample: videomme.VideoMMEQuestion) -> str:
    question = videomme._normalize_query_text(sample.question)
    option_texts: list[str] = []
    for option in sample.options:
        option_text = videomme._normalize_query_text(option)
        option_text = OPTION_LABEL_PATTERN.sub("", option_text).strip()
        if option_text:
            option_texts.append(option_text)
    options = " ".join(option_texts)
    return f"Question: {question} {options}".strip()


def _format_query_text(sample: videomme.VideoMMEQuestion, query_text: str) -> str:
    option_lines = "\n".join(f"  {option}" for option in sample.options)
    fields = [
        "Frame-selection query:",
        query_text,
        "Options:",
        option_lines,
    ]
    if sample.answer:
        fields.append(f"Answer: {sample.answer}")
    if sample.task_type:
        fields.append(f"Task type: {sample.task_type}")
    return "\n".join(fields)


def _wrap_text_preserving_lines(text: str, *, width: int) -> str:
    wrapped_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )
    return "\n".join(wrapped_lines)


def _resolve_output_path(output: Path) -> tuple[Path, bool]:
    output_path = _to_abs_path(output)
    if output_path is None:
        raise ValueError("Failed to resolve output path.")
    if output_path.suffix.lower() not in MATPLOTLIB_SUFFIXES:
        return output_path.with_suffix(".png"), True
    return output_path, False


def _metadata_float_list(metadata: dict[str, Any], key: str) -> list[float]:
    values = metadata.get(key) or []
    return [float(value) for value in values]


def _metadata_int_list(metadata: dict[str, Any], key: str) -> list[int]:
    values = metadata.get(key) or []
    return [int(value) for value in values]


def _plot_score_bars(
    *,
    ax: Any,
    x_values: list[int],
    scores: list[float],
    selected_positions: list[int],
    title: str,
    ylabel: str,
    bar_color: str,
) -> None:
    selected_set = set(selected_positions)
    colors = ["#d62728" if index in selected_set else bar_color for index in x_values]
    ax.bar(x_values, scores, color=colors, width=0.9)
    valid_selected_positions = [
        position for position in selected_positions if position < len(scores)
    ]
    if valid_selected_positions:
        ax.scatter(
            valid_selected_positions,
            [scores[position] for position in valid_selected_positions],
            s=72,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.8,
            label="selected",
            zorder=3,
        )
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("uniform candidate position")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(x_values) > 0:
        ax.set_xlim(-1, len(x_values))
    if valid_selected_positions:
        ax.legend(loc="upper right", frameon=False)


def _render_matplotlib_figure(
    *,
    plt: Any,
    output_path: Path,
    sample_index: int,
    sample: videomme.VideoMMEQuestion,
    query_text: str,
    uniform_selection: Any,
    frame_selection: Any,
    panel_cols: int,
) -> None:
    metadata = frame_selection.metadata
    gumbel_frames = frame_selection.frames.detach().cpu()
    uniform_frames = uniform_selection.frames.detach().cpu()
    selected_positions = [int(value) for value in metadata.get("selected_candidate_positions") or []]
    selected_ranked_positions = [
        int(value) for value in metadata.get("selected_ranked_candidate_positions") or []
    ]
    selected_original_indices = [
        int(value) for value in metadata.get("selected_original_indices") or []
    ]
    candidate_scores = _metadata_float_list(metadata, "candidate_frame_importance_scores")
    gumbel_final_scores = _metadata_float_list(metadata, "gumbel_noisy_scores")
    gumbel_noise = _metadata_float_list(metadata, "gumbel_noise")
    rank_by_position = {
        position: rank + 1 for rank, position in enumerate(selected_ranked_positions)
    }

    panel_cols = max(1, int(panel_cols))
    uniform_rows = max(1, math.ceil(int(uniform_frames.shape[0]) / panel_cols))
    gumbel_rows = max(1, math.ceil(int(gumbel_frames.shape[0]) / panel_cols))
    figure_width = max(12.0, panel_cols * 3.1)
    figure_height = 5.8 + (uniform_rows + gumbel_rows) * 3.0
    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=3 + uniform_rows + gumbel_rows,
        ncols=panel_cols,
        height_ratios=[1.2, 1.25, 1.25, *([2.5] * (uniform_rows + gumbel_rows))],
    )

    query_ax = fig.add_subplot(grid[0, :])
    query_ax.axis("off")
    query_title = (
        f"Video-MME Gumbel frame selection | sample_index={sample_index} | "
        f"video={sample.video_id} | question_id={sample.question_id}"
    )
    query_ax.set_title(query_title, loc="left", fontsize=13, fontweight="bold", pad=8)
    wrapped_query = _wrap_text_preserving_lines(
        _format_query_text(sample, query_text),
        width=140,
    )
    query_ax.text(
        0.0,
        0.92,
        wrapped_query,
        transform=query_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.25,
        family="monospace",
    )

    x_values = list(range(len(candidate_scores)))
    clip_ax = fig.add_subplot(grid[1, :])
    _plot_score_bars(
        ax=clip_ax,
        x_values=x_values,
        scores=candidate_scores,
        selected_positions=selected_positions,
        title="CLIP frame similarity before Gumbel sampling",
        ylabel="CLIP similarity",
        bar_color="#6f8fc5",
    )

    gumbel_ax = fig.add_subplot(grid[2, :])
    if len(gumbel_final_scores) == len(candidate_scores):
        _plot_score_bars(
            ax=gumbel_ax,
            x_values=x_values,
            scores=gumbel_final_scores,
            selected_positions=selected_positions,
            title=(
                "Final Gumbel score = log_softmax(CLIP / temperature) + noise "
                f"(temperature={metadata.get('gumbel_temperature')}, "
                f"gumbel_seed={metadata.get('gumbel_seed')})"
            ),
            ylabel="final score",
            bar_color="#7c6bb0",
        )
    else:
        gumbel_ax.axis("off")
        gumbel_ax.text(
            0.0,
            0.5,
            "Gumbel final scores are unavailable or length-mismatched.",
            transform=gumbel_ax.transAxes,
            va="center",
            ha="left",
            fontsize=10,
        )

    uniform_original_indices = _metadata_int_list(uniform_selection.metadata, "sampled_indices")
    for display_index, frame in enumerate(uniform_frames):
        row = 3 + display_index // panel_cols
        col = display_index % panel_cols
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(_frame_to_uint8(frame))
        ax.axis("off")
        original_index = (
            uniform_original_indices[display_index]
            if display_index < len(uniform_original_indices)
            else display_index
        )
        header = "Uniform sequence\n" if display_index == 0 else ""
        ax.set_title(
            f"{header}uniform {display_index} | orig {original_index}",
            fontsize=9,
        )

    total_uniform_panels = uniform_rows * panel_cols
    for empty_index in range(int(uniform_frames.shape[0]), total_uniform_panels):
        row = 3 + empty_index // panel_cols
        col = empty_index % panel_cols
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    gumbel_start_row = 3 + uniform_rows
    for display_index, frame in enumerate(gumbel_frames):
        row = gumbel_start_row + display_index // panel_cols
        col = display_index % panel_cols
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(_frame_to_uint8(frame))
        ax.axis("off")
        candidate_position = (
            selected_positions[display_index]
            if display_index < len(selected_positions)
            else display_index
        )
        original_index = (
            selected_original_indices[display_index]
            if display_index < len(selected_original_indices)
            else candidate_position
        )
        rank = rank_by_position.get(candidate_position, display_index + 1)
        frame_score = (
            candidate_scores[candidate_position]
            if candidate_position < len(candidate_scores)
            else None
        )
        final_score = (
            gumbel_final_scores[candidate_position]
            if candidate_position < len(gumbel_final_scores)
            else None
        )
        noise_score = (
            gumbel_noise[candidate_position]
            if candidate_position < len(gumbel_noise)
            else None
        )
        clip_text = "" if frame_score is None else f" | clip {frame_score:.4f}"
        final_text = "" if final_score is None else f"\nfinal {final_score:.4f}"
        noise_text = "" if noise_score is None else f" | noise {noise_score:.4f}"
        header = "Gumbel sequence\n" if display_index == 0 else ""
        ax.set_title(
            f"{header}sel {display_index} | rank {rank}\n"
            f"cand {candidate_position} | orig {original_index}{clip_text}"
            f"{final_text}{noise_text}",
            fontsize=9,
        )

    total_gumbel_panels = gumbel_rows * panel_cols
    for empty_index in range(int(gumbel_frames.shape[0]), total_gumbel_panels):
        row = gumbel_start_row + empty_index // panel_cols
        col = empty_index % panel_cols
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_metadata(
    output_path: Path,
    *,
    sample_index: int,
    sample: videomme.VideoMMEQuestion,
    query_text: str,
    uniform_selection: Any,
    frame_selection: Any,
) -> Path:
    metadata_path = output_path.with_suffix(".json")
    payload = {
        "sample_index": sample_index,
        "video_id": sample.video_id,
        "video_name": sample.video_name,
        "question_id": sample.question_id,
        "question": sample.question,
        "options": sample.options,
        "answer": sample.answer,
        "duration": sample.duration,
        "domain": sample.domain,
        "sub_category": sample.sub_category,
        "task_type": sample.task_type,
        "video_path": str(sample.video_path),
        "query_text": query_text,
        "uniform_selection_metadata": _json_safe(uniform_selection.metadata),
        "frame_selection_metadata": _json_safe(frame_selection.metadata),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def main() -> None:
    args = _parse_args()
    script_start = time.perf_counter()
    plt = _load_matplotlib()
    config = _load_runtime_config(args)

    with tempfile.TemporaryDirectory(prefix="videomme_gumbel_query_") as temp_name:
        temp_dir = Path(temp_name)
        _progress(args, "loading Video-MME samples")
        samples = _load_samples(args, config)
        sample_index, sample = _pick_sample(samples, args)
        query_text = _build_frame_selection_query_text(sample)
        query_file = temp_dir / "query.txt"
        query_file.write_text(query_text + "\n", encoding="utf-8")

        _apply_selector_overrides(config, args, query_file)
        OmegaConf.resolve(config)
        _progress(args, f"selected sample {sample_index}: {sample.video_id}/{sample.question_id}")
        _progress(args, f"video: {sample.video_path}")
        _progress(args, f"frame-selection query: {query_text}")

        frame_selector = instantiate(config.frame_selection)
        _progress(args, "running Gumbel frame selection")
        stage_start = time.perf_counter()
        frame_selection = frame_selector(video_path=str(sample.video_path))
        if frame_selection.frames is None:
            raise ValueError("Frame selector did not return frames.")
        _progress(
            args,
            f"selected {int(frame_selection.frames.shape[0])} frame(s) "
            f"from {frame_selection.metadata.get('candidate_num_frames')} candidates "
            f"in {time.perf_counter() - stage_start:.1f}s",
        )
        _progress(args, "sampling uniform baseline sequence")
        selector_config = config.frame_selection
        uniform_selection = uniform_sampling(
            video_path=str(sample.video_path),
            num_frames=int(selector_config.get("num_frames") or frame_selection.frames.shape[0]),
            max_side=selector_config.get("max_side"),
            ensure_qwen_compatibility=bool(
                selector_config.get("ensure_qwen_compatibility", True)
            ),
            qwen_factor=int(selector_config.get("qwen_factor") or 28),
        )

        output_path, coerced_output_suffix = _resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _render_matplotlib_figure(
            plt=plt,
            output_path=output_path,
            sample_index=sample_index,
            sample=sample,
            query_text=query_text,
            uniform_selection=uniform_selection,
            frame_selection=frame_selection,
            panel_cols=args.panel_cols,
        )
        metadata_path = _write_metadata(
            output_path,
            sample_index=sample_index,
            sample=sample,
            query_text=query_text,
            uniform_selection=uniform_selection,
            frame_selection=frame_selection,
        )

    print(f"Output      : {output_path}")
    if coerced_output_suffix:
        print("Note        : output suffix was changed to .png for matplotlib.")
    print(f"Metadata    : {metadata_path}")
    print(f"Sample      : {sample.video_id}/{sample.question_id} (index={sample_index})")
    print(f"Video       : {sample.video_path}")
    print(f"Query       : {query_text}")
    print(f"Question    : {sample.question}")
    print(f"Options     : {sample.options}")
    print(f"Selected    : {frame_selection.metadata.get('selected_original_indices')}")
    print(f"Elapsed     : {time.perf_counter() - script_start:.1f}s")

    if args.show:
        webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()
