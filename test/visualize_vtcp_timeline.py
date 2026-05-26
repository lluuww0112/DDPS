from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_SUFFIXES = {".html", ".htm"}
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VTCP 이동 평균 타임라인과 선택 프레임을 시각화합니다.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "vtcp.yaml",
        help="사용할 VTCP config 파일 경로",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="config의 invoke.video_path 대신 사용할 비디오 경로",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="config의 invoke.query_file 대신 사용할 질의 파일 경로",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default=None,
        help="질의를 문자열로 직접 지정할 때 사용",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test" / "artifacts" / "vtcp_timeline.html",
        help="인터랙티브 HTML 그래프 출력 파일 경로",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="저장 후 브라우저로도 열기",
    )
    return parser.parse_args()


def _load_plot_backend() -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit(
            "plotly가 필요합니다. 예: `pip install plotly`"
        ) from exc
    return go, make_subplots


def _resolve_runtime_config(args: argparse.Namespace) -> Any:
    config = OmegaConf.load(str(args.config))
    if args.video_path is not None:
        config.invoke.video_path = str(args.video_path)
    if args.query_file is not None:
        config.invoke.query_file = str(args.query_file)
        config.frame_selection.query_file = str(args.query_file)
    if args.query_text is not None:
        config.frame_selection.query_text = str(args.query_text)
    config.frame_selection.store_diagnostics = True
    return config


def _instantiate_selector(config: Any) -> Any:
    return instantiate(config.frame_selection)


def _build_selector_kwargs(config: Any) -> tuple[str, dict[str, Any]]:
    invoke_cfg = config.invoke
    video_path = Path(str(invoke_cfg.video_path)).expanduser()
    if not video_path.is_absolute():
        video_path = (REPO_ROOT / video_path).resolve()

    selector_kwargs: dict[str, Any] = {}
    query_text = config.frame_selection.get("query_text")
    if query_text is not None:
        selector_kwargs["query_text"] = str(query_text)
    return str(video_path), selector_kwargs


def _extract_plot_data(metadata: dict[str, Any]) -> dict[str, Any]:
    frame_indices = list(metadata.get("embedded_frame_indices") or [])
    if not frame_indices:
        raise ValueError(
            "VTCP metadata에 `embedded_frame_indices`가 없습니다. "
            "store_diagnostics=true로 실행되었는지 확인하세요."
        )

    smoothed_transition_scores = metadata.get("smoothed_transition_scores")
    raw_scores = metadata.get("raw_transition_scores")
    control_signal_scores = metadata.get("control_signal_scores")
    if control_signal_scores is None:
        control_signal_scores = smoothed_transition_scores
    if control_signal_scores is None:
        raise ValueError("VTCP metadata에 stride controller용 score가 없습니다.")

    visited_indices = list(metadata.get("visited_indices") or [])
    selected_indices = list(
        metadata.get("selected_original_indices")
        or metadata.get("sampled_indices")
        or []
    )
    stride_history = list(metadata.get("stride_history") or [])
    visited_control_scores = list(
        metadata.get("visited_control_scores")
        or metadata.get("visited_congestion_scores")
        or []
    )
    score_threshold = metadata.get("score_threshold")

    return {
        "frame_indices": [int(index) for index in frame_indices],
        "smoothed_transition_scores": (
            [float(value) for value in smoothed_transition_scores]
            if smoothed_transition_scores is not None
            else None
        ),
        "raw_scores": (
            [float(value) for value in raw_scores]
            if raw_scores is not None
            else None
        ),
        "control_signal_scores": (
            [float(value) for value in control_signal_scores]
            if control_signal_scores is not None
            else None
        ),
        "visited_indices": [int(index) for index in visited_indices],
        "selected_indices": [int(index) for index in selected_indices],
        "stride_history": [int(value) for value in stride_history],
        "visited_control_scores": [float(value) for value in visited_control_scores],
        "score_threshold": (
            float(score_threshold)
            if score_threshold is not None
            else None
        ),
    }


def _values_for_indices(
    reference_indices: list[int],
    values: list[float],
    target_indices: list[int],
) -> tuple[list[int], list[float]]:
    value_map = {
        int(index): float(value)
        for index, value in zip(reference_indices, values)
    }
    resolved_x: list[int] = []
    resolved_y: list[float] = []
    for index in target_indices:
        if int(index) in value_map:
            resolved_x.append(int(index))
            resolved_y.append(value_map[int(index)])
    return resolved_x, resolved_y


def _resolve_output_path(output_path: Path) -> tuple[Path, bool]:
    resolved_output = output_path.expanduser()
    if not resolved_output.is_absolute():
        resolved_output = (REPO_ROOT / resolved_output).resolve()

    coerced_to_html = resolved_output.suffix.lower() not in HTML_SUFFIXES
    if coerced_to_html:
        resolved_output = resolved_output.with_suffix(".html")
    return resolved_output, coerced_to_html


def _build_hover_template(
    *,
    include_raw: bool,
    include_smoothed_transition: bool,
) -> str:
    parts = [
        "frame=%{x}",
        "control_score=%{y:.4f}",
    ]
    if include_raw:
        parts.append("raw=%{customdata[0]:.4f}")
    if include_smoothed_transition:
        index = 1 if include_raw else 0
        parts.append(f"smoothed_transition=%{{customdata[{index}]:.4f}}")
    return "<br>".join(parts) + "<extra></extra>"


def _plot_timeline(
    *,
    go: Any,
    make_subplots: Any,
    plot_data: dict[str, Any],
    metadata: dict[str, Any],
    output_path: Path,
    show: bool,
) -> None:
    frame_indices = plot_data["frame_indices"]
    smoothed_transition_scores = plot_data["smoothed_transition_scores"]
    raw_scores = plot_data["raw_scores"]
    control_signal_scores = plot_data["control_signal_scores"]
    visited_indices = plot_data["visited_indices"]
    selected_indices = plot_data["selected_indices"]
    stride_history = plot_data["stride_history"]
    visited_control_scores = plot_data["visited_control_scores"]
    score_threshold = plot_data["score_threshold"]
    total_frames = int(metadata.get("total_frames") or (max(frame_indices) + 1))
    density_bin_size = max(1, (total_frames + 49) // 50)
    has_distinct_smoothed_reference = (
        smoothed_transition_scores is not None
        and len(smoothed_transition_scores) == len(frame_indices)
        and smoothed_transition_scores != control_signal_scores
    )
    x_line = [frame_indices[0], frame_indices[-1]] if frame_indices else [0, total_frames]

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.24, 0.16],
        specs=[
            [{}],
            [{"secondary_y": True}],
            [{}],
        ],
        subplot_titles=(
            "Temporal Transition Timeline",
            "Stride Controller Trace",
            "Sampling Density",
        ),
    )

    main_customdata = []
    for index, _ in enumerate(frame_indices):
        row: list[float | None] = []
        if raw_scores is not None and len(raw_scores) == len(frame_indices):
            row.append(float(raw_scores[index]))
        if has_distinct_smoothed_reference:
            row.append(float(smoothed_transition_scores[index]))
        main_customdata.append(row)

    if raw_scores is not None and len(raw_scores) == len(frame_indices):
        figure.add_trace(
            go.Scatter(
                x=frame_indices,
                y=raw_scores,
                mode="lines",
                name="Raw transition score",
                line={"color": "rgba(148, 163, 184, 0.9)", "width": 1.4},
                hovertemplate="frame=%{x}<br>raw=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if has_distinct_smoothed_reference:
        figure.add_trace(
            go.Scatter(
                x=frame_indices,
                y=smoothed_transition_scores,
                mode="lines",
                name="Smoothed transition score",
                line={"color": "rgba(37, 99, 235, 0.35)", "width": 1.8, "dash": "dot"},
                hovertemplate="frame=%{x}<br>smoothed_transition=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=frame_indices,
            y=control_signal_scores,
            mode="lines",
            name="Control signal score",
            line={"color": "#2563EB", "width": 2.8},
            customdata=main_customdata,
            hovertemplate=_build_hover_template(
                include_raw=raw_scores is not None and len(raw_scores) == len(frame_indices),
                include_smoothed_transition=has_distinct_smoothed_reference,
            ),
        ),
        row=1,
        col=1,
    )

    visited_x, visited_y = _values_for_indices(
        frame_indices,
        control_signal_scores,
        visited_indices,
    )
    selected_x, selected_y = _values_for_indices(
        frame_indices,
        control_signal_scores,
        selected_indices,
    )

    if visited_x:
        figure.add_trace(
            go.Scatter(
                x=visited_x,
                y=visited_y,
                mode="markers",
                name="Visited frames",
                marker={
                    "size": 9,
                    "color": "#F59E0B",
                    "line": {"color": "#7C2D12", "width": 0.8},
                },
                hovertemplate="visited frame=%{x}<br>control_score=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if selected_x:
        figure.add_trace(
            go.Scatter(
                x=selected_x,
                y=selected_y,
                mode="markers",
                name="Selected frames",
                marker={
                    "size": 16,
                    "color": "#DC2626",
                    "symbol": "star",
                    "line": {"color": "#111827", "width": 1.0},
                },
                hovertemplate="selected frame=%{x}<br>control_score=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    controller_positions = visited_indices[: len(stride_history)]
    if controller_positions and stride_history:
        figure.add_trace(
            go.Scatter(
                x=controller_positions,
                y=stride_history,
                mode="lines+markers",
                name="Stride history",
                line={"color": "#059669", "width": 2.2},
                marker={"size": 7, "color": "#059669"},
                hovertemplate="frame=%{x}<br>stride=%{y}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

    if controller_positions and visited_control_scores:
        figure.add_trace(
            go.Scatter(
                x=controller_positions,
                y=visited_control_scores[: len(controller_positions)],
                mode="lines+markers",
                name="Visited control score",
                line={"color": "#7C3AED", "width": 2.0, "dash": "dash"},
                marker={"size": 6, "color": "#7C3AED"},
                hovertemplate="frame=%{x}<br>control_score=%{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    if score_threshold is not None:
        threshold_label = f"Score threshold ({score_threshold:.4f})"
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=[score_threshold, score_threshold],
                mode="lines",
                name=threshold_label,
                line={"color": "#DC2626", "width": 1.8, "dash": "dash"},
                hovertemplate="score_threshold=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=[score_threshold, score_threshold],
                mode="lines",
                name=threshold_label,
                showlegend=False,
                line={"color": "#DC2626", "width": 1.6, "dash": "dash"},
                hovertemplate="score_threshold=%{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    if visited_indices:
        figure.add_trace(
            go.Histogram(
                x=visited_indices,
                xbins={"start": 0, "end": total_frames, "size": density_bin_size},
                marker={
                    "color": "rgba(245, 158, 11, 0.38)",
                    "line": {"color": "rgba(124, 45, 18, 0.35)", "width": 1},
                },
                name="Visited density",
                showlegend=False,
                hovertemplate="visited bin center=%{x}<br>count=%{y}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=visited_indices,
                y=[0.0] * len(visited_indices),
                mode="markers",
                name="Visited density strip",
                showlegend=False,
                marker={
                    "symbol": "line-ns-open",
                    "size": 16,
                    "color": "#F59E0B",
                    "line": {"color": "#92400E", "width": 1.2},
                },
                hovertemplate="visited frame=%{x}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    if selected_indices:
        figure.add_trace(
            go.Histogram(
                x=selected_indices,
                xbins={"start": 0, "end": total_frames, "size": density_bin_size},
                marker={
                    "color": "rgba(220, 38, 38, 0.32)",
                    "line": {"color": "rgba(17, 24, 39, 0.3)", "width": 1},
                },
                name="Selected density",
                showlegend=False,
                hovertemplate="selected bin center=%{x}<br>count=%{y}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=selected_indices,
                y=[0.0] * len(selected_indices),
                mode="markers",
                name="Selected density strip",
                showlegend=False,
                marker={
                    "symbol": "star",
                    "size": 11,
                    "color": "#DC2626",
                    "line": {"color": "#111827", "width": 1.0},
                },
                hovertemplate="selected frame=%{x}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    safe_query = html.escape(str(metadata.get("query_text", "")).strip()[:120])
    text_block = f"controller={metadata.get('stride_controller', 'unknown')}  "
    if score_threshold is not None:
        text_block += f"score_threshold={score_threshold:.4f}  "
    text_block += (
        f"embedded={metadata.get('embedded_frame_count')}  "
        f"visited={metadata.get('visited_frame_count')}  "
        f"selected={metadata.get('num_frames')}  "
        f"effective_stride={metadata.get('effective_embedding_stride')}"
    )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": f"VTCP Temporal Timeline<br><sup>query={safe_query}</sup>",
            "x": 0.5,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"l": 70, "r": 70, "t": 130, "b": 70},
        barmode="overlay",
        height=980,
    )
    figure.add_annotation(
        x=0.0,
        y=1.14,
        xref="paper",
        yref="paper",
        text=text_block,
        showarrow=False,
        align="left",
        font={"size": 12},
        bgcolor="rgba(255, 255, 255, 0.88)",
        bordercolor="rgba(148, 163, 184, 0.8)",
        borderwidth=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.update_xaxes(
        title_text="Original frame index",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    figure.update_yaxes(
        title_text="Smoothed transition / control score",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Stride",
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Control score",
        showgrid=False,
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.update_yaxes(
        title_text="Count",
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=True,
        zerolinecolor="rgba(148, 163, 184, 0.28)",
        row=3,
        col=1,
    )
    figure.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=show,
    )


def main() -> None:
    args = _parse_args()
    go, make_subplots = _load_plot_backend()
    config = _resolve_runtime_config(args)
    selector = _instantiate_selector(config)
    video_path, selector_kwargs = _build_selector_kwargs(config)

    result = selector(video_path=video_path, **selector_kwargs)
    metadata = dict(result.metadata)
    plot_data = _extract_plot_data(metadata)
    output_path, coerced_to_html = _resolve_output_path(args.output)

    _plot_timeline(
        go=go,
        make_subplots=make_subplots,
        plot_data=plot_data,
        metadata=metadata,
        output_path=output_path,
        show=args.show,
    )

    if coerced_to_html:
        print("Requested a non-HTML output path, so the Plotly report was saved as HTML instead.")
    print(f"Saved plot: {output_path}")
    print(f"Visited frames : {plot_data['visited_indices']}")
    print(f"Selected frames: {plot_data['selected_indices']}")


if __name__ == "__main__":
    main()
