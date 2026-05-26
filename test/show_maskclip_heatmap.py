from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SVG_SUFFIXES = {".svg"}
DATASET_CHOICES = ("direct", "nextqa", "mvbench", "videomme", "egoschema")
DEFAULT_PATCH_SELECTION_KEY = "patch_selection_v4"
DEFAULT_CLIP_LARGE_MODEL = "openai/clip-vit-large-patch14"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.PatchSelection.DenseDPS.selection_v1 import _load_queries
import test.show_selected_patch as selected_patch_viz
from test.show_selected_patch import (
    _apply_common_config_overrides,
    _configure_query_file,
    _display_frame_indices,
    _frame_to_uint8,
    _frames_for_temporal_grid,
    _resolve_target,
    _run_clip_only_selection_diagnostics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render MaskCLIP score maps as matplotlib SVG heatmaps.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "DDPS.yaml",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--patch-selection-key",
        type=str,
        default=DEFAULT_PATCH_SELECTION_KEY,
        help="Config key such as patch_selection_v4 or patch_selection_v5.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="direct",
        help="Use a direct video path or a supported eval dataset sample.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Override config.invoke.video_path for direct mode.",
    )
    parser.add_argument("--query-file", type=Path, default=None)
    parser.add_argument("--query-text", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--uid-map-file", type=Path, default=None)
    parser.add_argument("--video-map-file", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test" / "artifacts" / "maskclip_heatmap.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--score-grid",
        choices=("merged", "raw"),
        default="raw",
        help="Visualize Qwen-merged token scores or raw CLIP patch scores.",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default=DEFAULT_CLIP_LARGE_MODEL,
        help="CLIP model used for MaskCLIP scoring.",
    )
    parser.add_argument(
        "--layout",
        choices=("comparison", "panels"),
        default="comparison",
        help="Render a single original-vs-heatmap comparison or multi-frame panels.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=3,
        help="Temporal grid frame index for comparison layout.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Visualize a single query index instead of the configured aggregation.",
    )
    parser.add_argument(
        "--max-frame-panels",
        type=int,
        default=24,
        help="Maximum number of temporal panels to render.",
    )
    parser.add_argument(
        "--panel-cols",
        type=int,
        default=4,
        help="Number of panel columns.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.72,
        help="Heatmap alpha when overlaid on video frames.",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Do not draw the sampled video frame below the heatmap.",
    )
    parser.add_argument(
        "--hide-selected",
        action="store_true",
        help="Do not outline selected patches.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=2.0,
        help="Lower percentile for automatic color scaling.",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=98.0,
        help="Upper percentile for automatic color scaling.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=None,
        help="Override patch selector batch_size for this run.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress progress logs.",
    )
    parser.add_argument("--show", action="store_true", help="Open the SVG after writing.")
    return parser.parse_args()


def _progress(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "quiet_progress", False):
        print(f"[show_maskclip_heatmap] {message}", flush=True)


def _load_matplotlib_backend() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with `pip install matplotlib`."
        ) from exc
    return plt, patches


def _resolve_output_path(output_path: Path) -> tuple[Path, bool]:
    resolved = output_path.expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    coerced = resolved.suffix.lower() not in SVG_SUFFIXES
    if coerced:
        resolved = resolved.with_suffix(".svg")
    return resolved, coerced


def _target_args_from_config(config: Any, args: argparse.Namespace) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    if resolved.dataset != "direct":
        return resolved
    if resolved.query_file is None and resolved.query_text is None:
        invoke_cfg = config.get("invoke") or {}
        configured_query_file = invoke_cfg.get("query_file")
        if configured_query_file:
            resolved.query_file = Path(str(configured_query_file))
    return resolved


def _apply_heatmap_overrides(config: Any, args: argparse.Namespace) -> None:
    selector = config.vlm.get("patch_selector")
    if selector is not None and args.clip_model_name:
        selector.clip_model_name = str(args.clip_model_name)


def _aggregate_query_scores(
    query_scores: torch.Tensor,
    *,
    aggregation: str,
    query_index: int | None,
) -> tuple[torch.Tensor, str]:
    query_count = int(query_scores.shape[1])
    if query_index is not None:
        if query_index < 0 or query_index >= query_count:
            raise IndexError(
                f"--query-index must be in [0, {query_count - 1}], got {query_index}."
            )
        return query_scores[:, query_index], f"query {query_index}"

    normalized = str(aggregation).strip().lower()
    if normalized == "max":
        return query_scores.max(dim=1).values, "query max"
    if normalized == "mean":
        return query_scores.mean(dim=1), "query mean"
    raise ValueError(f"Unsupported aggregation: {aggregation}. Use max or mean.")


def _resolve_score_data(
    diagnostics: Any,
    *,
    score_grid: str,
    aggregation: str,
    query_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
    if score_grid == "raw":
        query_scores = diagnostics.raw_clip_query_scores
        selected_mask = diagnostics.raw_clip_selected_mask
        metadata = diagnostics.raw_clip_metadata
        grid_label = "raw CLIP patch grid"
    else:
        query_scores = diagnostics.query_scores
        selected_mask = diagnostics.selected_mask
        metadata = diagnostics.selector_metadata
        grid_label = "merged Qwen token grid"

    if query_scores is None:
        raise ValueError("MaskCLIP query score diagnostics are unavailable.")
    if selected_mask is None:
        raise ValueError("Selected patch mask is unavailable.")
    if metadata is None:
        raise ValueError("Selector metadata is unavailable.")

    scores, aggregation_label = _aggregate_query_scores(
        query_scores.detach().cpu().to(dtype=torch.float32),
        aggregation=aggregation,
        query_index=query_index,
    )
    return scores, selected_mask.detach().cpu(), metadata, f"{grid_label}, {aggregation_label}"


def _auto_color_limits(
    scores: torch.Tensor,
    *,
    vmin: float | None,
    vmax: float | None,
    percentile_min: float,
    percentile_max: float,
) -> tuple[float, float]:
    flat = scores.detach().cpu().flatten()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        raise ValueError("Heatmap scores do not contain finite values.")

    low = float(vmin) if vmin is not None else float(
        torch.quantile(finite, max(0.0, min(100.0, percentile_min)) / 100.0).item()
    )
    high = float(vmax) if vmax is not None else float(
        torch.quantile(finite, max(0.0, min(100.0, percentile_max)) / 100.0).item()
    )
    if math.isclose(low, high):
        epsilon = 1e-6 if math.isclose(low, 0.0) else abs(low) * 1e-6
        low -= epsilon
        high += epsilon
    if low > high:
        raise ValueError(f"Invalid color limits: vmin={low}, vmax={high}.")
    return low, high


def _draw_selected_rectangles(
    *,
    ax: Any,
    patches: Any,
    selected_mask: torch.Tensor,
    frame_idx: int,
    width: int,
    height: int,
) -> None:
    _, grid_h, grid_w = selected_mask.shape
    cell_w = width / float(grid_w)
    cell_h = height / float(grid_h)
    for row in range(grid_h):
        for col in range(grid_w):
            if not bool(selected_mask[frame_idx, row, col].item()):
                continue
            ax.add_patch(
                patches.Rectangle(
                    (col * cell_w, row * cell_h),
                    cell_w,
                    cell_h,
                    fill=False,
                    edgecolor="#00e5ff",
                    linewidth=0.65,
                    alpha=0.95,
                )
            )


def _frame_score_stats(
    score_map: torch.Tensor,
    selected_mask: torch.Tensor,
) -> tuple[float, float, float]:
    flat_scores = score_map.detach().cpu().flatten().to(dtype=torch.float32)
    flat_mask = selected_mask.detach().cpu().flatten().to(dtype=torch.bool)
    avg_score = float(flat_scores.mean().item())
    if bool(flat_mask.any().item()):
        in_score = float(flat_scores[flat_mask].mean().item())
    else:
        in_score = float("nan")
    if bool((~flat_mask).any().item()):
        out_score = float(flat_scores[~flat_mask].mean().item())
    else:
        out_score = float("nan")
    return avg_score, in_score, out_score


def _single_frame_query_text(
    *,
    target: Any,
    queries: list[str],
    query_index: int | None,
) -> str:
    if query_index is not None and 0 <= query_index < len(queries):
        return queries[query_index]
    return target.query_text.strip()


def _plot_comparison(
    *,
    plt: Any,
    diagnostics: Any,
    target: Any,
    scores: torch.Tensor,
    selected_mask: torch.Tensor,
    metadata: dict[str, Any],
    queries: list[str],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    frame_count, grid_h, grid_w = [int(value) for value in scores.shape]
    frame_idx = int(args.frame_index)
    if frame_idx < 0 or frame_idx >= frame_count:
        raise IndexError(f"--frame-index must be in [0, {frame_count - 1}], got {frame_idx}.")

    display_frames, sampled_indices = _frames_for_temporal_grid(
        diagnostics.frame_selection,
        metadata,
    )
    frame = _frame_to_uint8(display_frames[frame_idx])
    score_map = scores[frame_idx]
    vmin, vmax = _auto_color_limits(
        score_map,
        vmin=args.vmin,
        vmax=args.vmax,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
    )
    avg_score, in_score, out_score = _frame_score_stats(
        score_map,
        selected_mask[frame_idx],
    )
    query_text = _single_frame_query_text(
        target=target,
        queries=queries,
        query_index=args.query_index,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.6),
        gridspec_kw={"width_ratios": [1.05, 1.16]},
        constrained_layout=True,
    )
    original_ax, heatmap_ax = axes
    original_ax.imshow(frame)
    original_ax.set_title(f"original frame {frame_idx} (sample={sampled_indices[frame_idx]})", fontsize=12)
    original_ax.set_xticks([])
    original_ax.set_yticks([])

    heatmap = heatmap_ax.imshow(
        score_map.detach().cpu().numpy(),
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    heatmap_ax.set_xlabel("patch_x", fontsize=12)
    heatmap_ax.set_ylabel("patch_y", fontsize=12)
    heatmap_ax.set_title(
        f'text: "{query_text}"\n'
        f"avg:{avg_score:.3f}\n"
        f"in:{in_score:.3f} / out:{out_score:.3f}",
        fontsize=15,
    )
    heatmap_ax.set_xlim(-0.5, grid_w - 0.5)
    heatmap_ax.set_ylim(grid_h - 0.5, -0.5)
    colorbar = fig.colorbar(heatmap, ax=heatmap_ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    *,
    plt: Any,
    patches: Any,
    diagnostics: Any,
    target: Any,
    scores: torch.Tensor,
    selected_mask: torch.Tensor,
    metadata: dict[str, Any],
    score_label: str,
    queries: list[str],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    frame_count, grid_h, grid_w = [int(value) for value in scores.shape]
    frame_indices = _display_frame_indices(frame_count, args.max_frame_panels)
    if not frame_indices:
        raise ValueError("No frame panels selected for rendering.")

    panel_cols = max(1, int(args.panel_cols))
    panel_cols = min(panel_cols, len(frame_indices))
    panel_rows = math.ceil(len(frame_indices) / panel_cols)
    panel_width = 3.1 if args.heatmap_only else 3.35
    panel_height = 2.75 if args.heatmap_only else 3.0
    fig_width = max(5.0, panel_cols * panel_width)
    fig_height = max(3.2, panel_rows * panel_height + 0.9)

    fig, axes = plt.subplots(
        panel_rows,
        panel_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    vmin, vmax = _auto_color_limits(
        scores,
        vmin=args.vmin,
        vmax=args.vmax,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
    )
    display_frames, sampled_indices = _frames_for_temporal_grid(
        diagnostics.frame_selection,
        metadata,
    )

    image_artist = None
    for panel_idx, frame_idx in enumerate(frame_indices):
        ax = axes[panel_idx // panel_cols][panel_idx % panel_cols]
        score_map = scores[frame_idx].numpy()
        title_parts = [f"t={frame_idx}", f"sample={sampled_indices[frame_idx]}"]

        if args.heatmap_only:
            height, width = grid_h, grid_w
        else:
            frame = _frame_to_uint8(display_frames[frame_idx])
            height, width = int(frame.shape[0]), int(frame.shape[1])
            ax.imshow(frame, extent=(0, width, height, 0), interpolation="nearest")

        image_artist = ax.imshow(
            score_map,
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=1.0 if args.heatmap_only else float(args.alpha),
            extent=(0, width, height, 0),
            interpolation="nearest",
        )
        if not args.hide_selected:
            _draw_selected_rectangles(
                ax=ax,
                patches=patches,
                selected_mask=selected_mask,
                frame_idx=frame_idx,
                width=width,
                height=height,
            )
            selected_count = int(selected_mask[frame_idx].sum().item())
            title_parts.append(f"selected={selected_count}")
        ax.set_title(" | ".join(title_parts), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for panel_idx in range(len(frame_indices), panel_rows * panel_cols):
        axes[panel_idx // panel_cols][panel_idx % panel_cols].axis("off")

    if image_artist is not None:
        colorbar = fig.colorbar(
            image_artist,
            ax=axes.ravel().tolist(),
            shrink=0.82,
            pad=0.012,
        )
        colorbar.set_label("MaskCLIP cosine score", fontsize=9)

    query_text = target.query_text.strip()
    if args.query_index is not None and 0 <= args.query_index < len(queries):
        query_text = queries[args.query_index]
    fig.suptitle(
        f"MaskCLIP heatmap ({score_label})\n{target.sample_label} | {query_text}",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    script_start = time.perf_counter()
    plt, patches = _load_matplotlib_backend()
    _progress(args, f"loading config: {args.config}")
    config = OmegaConf.load(str(args.config))
    _apply_common_config_overrides(config, args)
    _apply_heatmap_overrides(config, args)
    output_path, coerced_to_svg = _resolve_output_path(args.output)

    with tempfile.TemporaryDirectory(prefix="show_maskclip_heatmap_") as temp_name:
        temp_dir = Path(temp_name)
        _progress(args, "resolving visualization target")
        target_args = _target_args_from_config(config, args)
        target = _resolve_target(config=config, args=target_args, temp_dir=temp_dir)
        _configure_query_file(config, target.query_file)
        _progress(args, f"target video={target.video_path}")
        _progress(args, f"target query={target.query_text}")

        diagnostics_args = argparse.Namespace(**vars(args), skip_query_diagnostics=False)
        selected_patch_viz._progress = _progress
        diagnostics = _run_clip_only_selection_diagnostics(
            config=config,
            args=diagnostics_args,
            target=target,
        )
        selector_cfg = config.vlm.get("patch_selector") or {}
        aggregation = str(selector_cfg.get("aggregation") or "max")
        scores, selected_mask, metadata, score_label = _resolve_score_data(
            diagnostics,
            score_grid=args.score_grid,
            aggregation=aggregation,
            query_index=args.query_index,
        )
        queries = _load_queries(str(selector_cfg.get("query_file")))

        _progress(args, f"rendering matplotlib SVG: {output_path}")
        if args.layout == "comparison":
            _plot_comparison(
                plt=plt,
                diagnostics=diagnostics,
                target=target,
                scores=scores,
                selected_mask=selected_mask,
                metadata=metadata,
                queries=queries,
                args=args,
                output_path=output_path,
            )
        else:
            _plot_heatmaps(
                plt=plt,
                patches=patches,
                diagnostics=diagnostics,
                target=target,
                scores=scores,
                selected_mask=selected_mask,
                metadata=metadata,
                score_label=score_label,
                queries=queries,
                args=args,
                output_path=output_path,
            )

    _progress(args, f"done in {time.perf_counter() - script_start:.1f}s")
    print(f"Output      : {output_path}")
    if coerced_to_svg:
        print("Note        : output suffix was changed to .svg")
    print(f"Dataset     : {args.dataset}")
    print(f"Sample      : {target.sample_label}")
    print(f"Video       : {target.video_path}")
    print(f"Query       : {target.query_text}")
    print(f"Score grid  : {args.score_grid}")
    print(f"CLIP model  : {config.vlm.patch_selector.get('clip_model_name')}")
    print(f"Layout      : {args.layout}")
    if args.layout == "comparison":
        print(f"Frame index : {args.frame_index}")
    print(f"Selected    : {int(selected_mask.sum().item())}/{int(selected_mask.numel())}")
    if args.show:
        webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()
