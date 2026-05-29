from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from model.invoke import build_vlm, suppress_model_loading_output
from eval.runtime_metrics import (
    extract_runtime_metrics,
    format_runtime_summary_lines,
    init_runtime_metric_totals,
    summarize_runtime_metric_totals,
    update_runtime_metric_totals,
)


VIDEO_DIR_CANDIDATES = ("data", "videos", "video")
ANNOTATION_DIR_CANDIDATES = ("json", "annotations")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAILING_TIME_RANGE_PATTERN = re.compile(r"_(?:\d+(?:\.\d+)?)_(?:\d+(?:\.\d+)?)$")
INDEX_PATTERN = re.compile(r"\b(?:answer|option|choice)?\s*[:#\-]?\s*([0-9])\b", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b(?:answer|option|choice)?\s*[:#\-]?\s*([A-E])\b", re.IGNORECASE)


@dataclass(slots=True)
class MVBenchSample:
    task_name: str
    sample_index: int
    video_name: str
    question: str
    candidates: list[str]
    answer_text: str
    answer_index: int | None
    start: float | None
    end: float | None
    video_path: Path
    raw_item: dict[str, Any]


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _to_abs_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(to_absolute_path(str(path_value))).expanduser().resolve()


def _resolve_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    return resolved if resolved >= 0 else None


def _resolve_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    return resolved if resolved >= 0 else None


def _resolve_experiment_config_path(experiment_value: Any) -> Path:
    if experiment_value is None or not str(experiment_value).strip():
        raise ValueError(
            "`experiment` must be provided. "
            "Example: `python -m eval.mvbench experiment=base` or `experiment=trips`."
        )

    raw_value = str(experiment_value).strip()
    raw_path = Path(raw_value).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        candidates.append((CONFIG_DIR / raw_path).resolve())
        if raw_path.suffix != ".yaml":
            candidates.append((Path.cwd() / f"{raw_value}.yaml").resolve())
            candidates.append((CONFIG_DIR / f"{raw_value}.yaml").resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find experiment config for `{raw_value}`. "
        f"Tried: {[str(candidate) for candidate in candidates]}"
    )


def _normalize_match_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[^0-9a-z\s]+", " ", lowered)
    return " ".join(lowered.split())


def _find_named_path(search_root: Path, *, names: tuple[str, ...], expect_dir: bool) -> Path | None:
    candidate_roots = [
        search_root,
        search_root / "MVBench",
        search_root / "mvbench",
    ]
    for root in candidate_roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate.is_dir() == expect_dir:
                return candidate.resolve()

    for name in names:
        try:
            match = next(search_root.rglob(name))
        except StopIteration:
            match = None
        if match is not None and match.is_dir() == expect_dir:
            return match.resolve()
    return None


def _resolve_dataset_layout(eval_config: DictConfig | None) -> tuple[Path, Path, Path]:
    dataset_root = _to_abs_path(
        str(eval_config.get("dataset_root")) if eval_config and eval_config.get("dataset_root") else "./MVBench"
    )
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "MVBench dataset root could not be found. "
            "Set `mvbench.dataset_root` to the dataset directory."
        )

    annotations_dir = _to_abs_path(
        str(eval_config.get("annotations_dir")) if eval_config and eval_config.get("annotations_dir") else None
    )
    if annotations_dir is None:
        annotations_dir = _find_named_path(dataset_root, names=ANNOTATION_DIR_CANDIDATES, expect_dir=True)
    if annotations_dir is None:
        raise FileNotFoundError(
            "Could not find MVBench annotation directory. "
            f"Looked for: {ANNOTATION_DIR_CANDIDATES}"
        )

    data_dir = _to_abs_path(
        str(eval_config.get("data_dir")) if eval_config and eval_config.get("data_dir") else None
    )
    if data_dir is None:
        data_dir = _find_named_path(dataset_root, names=VIDEO_DIR_CANDIDATES, expect_dir=True)
    if data_dir is None:
        raise FileNotFoundError(
            "Could not find MVBench data directory. "
            f"Looked for: {VIDEO_DIR_CANDIDATES}"
        )

    return dataset_root, annotations_dir, data_dir


def _load_prompt_template(prompt_file: Path) -> dict[str, str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    sections = {"system": [], "user": []}
    current_section: str | None = None
    for line in prompt_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[SYSTEM]":
            current_section = "system"
            continue
        if stripped == "[USER]":
            current_section = "user"
            continue
        if current_section is not None:
            sections[current_section].append(line)

    user_template = "\n".join(sections["user"]).strip()
    if not user_template:
        raise ValueError(f"`[USER]` section is missing in prompt file: {prompt_file}")

    return {
        "system": "\n".join(sections["system"]).strip(),
        "user": user_template,
    }


def _collect_annotation_files(annotations_dir: Path, task_names: list[str] | None) -> list[Path]:
    requested = {name.strip() for name in (task_names or []) if str(name).strip()}
    files = sorted(path.resolve() for path in annotations_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No MVBench annotation json files were found under: {annotations_dir}")
    if not requested:
        return files
    selected = [path for path in files if path.stem in requested]
    missing = sorted(requested - {path.stem for path in selected})
    if missing:
        raise FileNotFoundError(f"Could not find MVBench task files for: {missing}")
    return selected


def _lookup_key_variants(path: Path, search_root: Path) -> set[str]:
    keys = {path.name.casefold()}
    if path.is_file():
        keys.add(path.stem.casefold())
        stripped_time_range = TRAILING_TIME_RANGE_PATTERN.sub("", path.stem)
        if stripped_time_range != path.stem:
            keys.add(stripped_time_range.casefold())

    try:
        relative_path = path.relative_to(search_root)
    except ValueError:
        relative_path = path
    relative_key = relative_path.as_posix()
    keys.add(relative_key.casefold())
    if path.is_file():
        relative_stem_key = relative_path.with_suffix("").as_posix()
        keys.add(relative_stem_key.casefold())
        stripped_relative_stem = TRAILING_TIME_RANGE_PATTERN.sub("", relative_stem_key)
        if stripped_relative_stem != relative_stem_key:
            keys.add(stripped_relative_stem.casefold())
    return {key for key in keys if key}


def _add_lookup_path(lookup: dict[str, list[Path]], key: str, path: Path) -> None:
    paths = lookup.setdefault(key, [])
    resolved = path.resolve()
    if resolved not in paths:
        paths.append(resolved)


def _collect_video_search_roots(data_dir: Path) -> list[Path]:
    candidates = [
        data_dir,
        data_dir.parent / "video",
        data_dir.parent / "videos",
    ]
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        roots.append(resolved)
        seen.add(resolved)
    return roots


def _build_video_lookup(data_dir: Path) -> tuple[dict[str, list[Path]], set[str]]:
    lookup: dict[str, list[Path]] = {}
    frame_dirs: set[Path] = set()
    search_roots = _collect_video_search_roots(data_dir)
    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                for key in _lookup_key_variants(path, search_root):
                    _add_lookup_path(lookup, key, path)
                continue
            if suffix in IMAGE_SUFFIXES:
                frame_dirs.add(path.parent)

        for frame_dir in frame_dirs:
            try:
                frame_dir.relative_to(search_root)
            except ValueError:
                continue
            for key in _lookup_key_variants(frame_dir, search_root):
                _add_lookup_path(lookup, key, frame_dir)

    if not lookup:
        raise FileNotFoundError(
            "No supported video files or frame directories were found under: "
            f"{[str(root) for root in search_roots] or [str(data_dir)]}"
        )
    duplicates = {key for key, paths in lookup.items() if len(paths) > 1}
    return lookup, duplicates


def _format_candidate_paths(candidates: list[Path], *, limit: int = 5) -> str:
    rendered = [str(path) for path in candidates[:limit]]
    if len(candidates) > limit:
        rendered.append(f"... and {len(candidates) - limit} more")
    return ", ".join(rendered)


def _resolve_video_path(
    video_name: str,
    video_lookup: dict[str, list[Path]],
    *,
    task_name: str,
) -> Path:
    candidates = video_lookup.get(video_name.casefold())
    if candidates is None:
        candidates = video_lookup.get(Path(video_name).stem.casefold())
    if not candidates:
        raise FileNotFoundError(f"Could not find MVBench video file for `{video_name}`.")
    candidates = sorted(candidates, key=lambda path: (len(path.parts), str(path).casefold()))
    if len(candidates) > 1:
        raise FileNotFoundError(
            "MVBench video lookup is ambiguous. "
            f"task={task_name}, video={video_name}, candidates={_format_candidate_paths(candidates)}"
        )
    return candidates[0]


def _extract_answer_index(answer_text: str, candidates: list[str]) -> int | None:
    normalized_answer = _normalize_match_text(answer_text)
    for index, candidate in enumerate(candidates):
        if _normalize_match_text(candidate) == normalized_answer:
            return index
    return None


def _load_samples(
    *,
    annotation_files: list[Path],
    video_lookup: dict[str, list[Path]],
    duplicate_keys: set[str],
    skip_missing_videos: bool = False,
) -> tuple[list[MVBenchSample], int, int]:
    samples: list[MVBenchSample] = []
    skipped = 0
    missing = 0
    for annotation_file in annotation_files:
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"MVBench task file must be a list: {annotation_file}")

        task_name = annotation_file.stem
        for sample_index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(f"MVBench sample must be an object: task={task_name}, index={sample_index}")
            video_name = str(item.get("video", "")).strip()
            question = str(item.get("question", "")).strip()
            candidates = item.get("candidates")
            answer_text = str(item.get("answer", "")).strip()
            if not video_name or not question or not isinstance(candidates, list) or not answer_text:
                raise ValueError(f"Invalid MVBench sample in {annotation_file}: {item}")
            name_key = video_name.casefold()
            stem_key = Path(video_name).stem.casefold()
            if name_key in duplicate_keys or stem_key in duplicate_keys:
                skipped += 1
                continue
            candidates_text = [str(candidate).strip() for candidate in candidates]
            try:
                video_path = _resolve_video_path(
                    video_name,
                    video_lookup,
                    task_name=task_name,
                )
            except FileNotFoundError:
                if skip_missing_videos:
                    missing += 1
                    continue
                raise
            samples.append(
                MVBenchSample(
                    task_name=task_name,
                    sample_index=sample_index,
                    video_name=video_name,
                    question=question,
                    candidates=candidates_text,
                    answer_text=answer_text,
                    answer_index=_extract_answer_index(answer_text, candidates_text),
                    start=_resolve_optional_float(item.get("start")),
                    end=_resolve_optional_float(item.get("end")),
                    video_path=video_path,
                    raw_item=dict(item),
                )
            )
    return samples, skipped, missing


def _build_option_block(candidates: list[str]) -> str:
    rows: list[str] = []
    for index, candidate in enumerate(candidates):
        label = chr(ord("A") + index)
        rows.append(f"{label}. {candidate}")
    return "\n".join(rows)


def _render_prompt(*, prompt_template: dict[str, str], sample: MVBenchSample) -> dict[str, str]:
    option_block = _build_option_block(sample.candidates)
    query_text = "\n".join(
        [
            "Select the best answer to the following multiple-choice question based on the video clip.",
            "Respond with only the letter of the correct option.",
            f"Question: {sample.question}",
            option_block,
            "The best answer is:",
        ]
    )
    format_values = _SafeFormatDict(
        {
            "task_name": sample.task_name,
            "video": sample.video_name,
            "question": sample.question,
            "query": query_text,
            "options": option_block,
            "option_block": option_block,
            "candidate_answers": option_block,
            "start": sample.start if sample.start is not None else "",
            "end": sample.end if sample.end is not None else "",
            **{f"option_{index}": option for index, option in enumerate(sample.candidates)},
        }
    )
    return {
        "system": prompt_template["system"].format_map(format_values).strip(),
        "user": prompt_template["user"].format_map(format_values).strip(),
    }


def _configure_dynamic_query_file(vlm: Any, query_file_path: Path) -> tuple[bool, tuple[str, ...]]:
    updated_targets: list[str] = []
    target_specs = (
        ("frame_selector", getattr(vlm, "frame_selector", None)),
        ("patch_selector", getattr(vlm, "patch_selector", None)),
    )

    for target_name, target in target_specs:
        if target is None:
            continue
        keywords = getattr(target, "keywords", None)
        if isinstance(keywords, dict) and "query_file" in keywords:
            keywords["query_file"] = str(query_file_path)
            updated_targets.append(target_name)

    return bool(updated_targets), tuple(updated_targets)


def _option_letter(index: int | None) -> str | None:
    if index is None or index < 0:
        return None
    return chr(ord("A") + index)


def _parse_prediction(response: str, candidates: list[str]) -> tuple[str | None, int | None, str]:
    stripped = response.strip()
    if not stripped:
        return None, None, "empty_response"

    for match in LETTER_PATTERN.finditer(stripped.upper()):
        index = ord(match.group(1)) - ord("A")
        if 0 <= index < len(candidates):
            return _option_letter(index), index, "letter_regex"

    for match in INDEX_PATTERN.finditer(stripped):
        index = int(match.group(1))
        if 0 <= index < len(candidates):
            return _option_letter(index), index, "index_regex"
        if 1 <= index <= len(candidates):
            adjusted = index - 1
            return _option_letter(adjusted), adjusted, "one_based_index_regex"

    normalized_response = _normalize_match_text(stripped)
    for index, candidate in enumerate(candidates):
        normalized_candidate = _normalize_match_text(candidate)
        if normalized_candidate and (
            normalized_candidate in normalized_response or normalized_response in normalized_candidate
        ):
            return _option_letter(index), index, "substring_match"

    return None, None, "raw_response"


def _write_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _natural_path_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return [int(part) if part.isdigit() else part for part in parts]


def _collect_frame_files(frame_dir: Path) -> list[Path]:
    return sorted(
        (path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_path_key,
    )


def _infer_frame_directory_fps(frame_dir: Path) -> float:
    for part in frame_dir.parts:
        match = re.search(r"fps([0-9]+(?:\.[0-9]+)?)", part.casefold())
        if match:
            fps = float(match.group(1))
            if fps > 0:
                return fps
    return 3.0


def _materialize_frame_directory(frame_dir: Path, temp_dir: Path) -> Path:
    frame_files = _collect_frame_files(frame_dir)
    if not frame_files:
        raise FileNotFoundError(f"No image frames were found under MVBench frame directory: {frame_dir}")

    path_hash = hashlib.sha1(str(frame_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    output_path = temp_dir / f"{frame_dir.name}_{path_hash}.mp4"
    if output_path.exists():
        return output_path

    first_frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read MVBench frame: {frame_files[0]}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, _infer_frame_directory_fps(frame_dir), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create MVBench frame-directory video writer: {output_path}")
    try:
        writer.write(first_frame)
        for frame_file in frame_files[1:]:
            frame = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read MVBench frame: {frame_file}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to create MVBench video from frame directory: {frame_dir}")
    return output_path


def _extract_video_clip(video_path: Path, start: float | None, end: float | None, temp_dir: Path) -> Path:
    if video_path.is_dir():
        materialized_video_path = _materialize_frame_directory(video_path, temp_dir)
        if start is None and end is None:
            return materialized_video_path
        return _extract_video_clip(materialized_video_path, start, end, temp_dir)

    if start is None and end is None:
        return video_path

    clip_start = max(float(start or 0.0), 0.0)
    clip_end = float(end) if end is not None else None
    if clip_end is not None and clip_end <= clip_start:
        return video_path

    ffmpeg_path = shutil.which("ffmpeg")
    clip_output = temp_dir / f"{video_path.stem}_{clip_start:.2f}_{(clip_end or -1):.2f}.mp4"

    if ffmpeg_path is not None:
        command = [ffmpeg_path, "-y", "-loglevel", "error", "-ss", f"{clip_start:.3f}", "-i", str(video_path)]
        if clip_end is not None:
            command.extend(["-to", f"{clip_end:.3f}"])
        command.extend(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_output)])
        try:
            subprocess.run(command, check=True)
            if clip_output.exists():
                return clip_output
        except (OSError, subprocess.CalledProcessError):
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open MVBench video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid MVBench clip size: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_output), fourcc, fps, (width, height))
    start_frame = int(round(clip_start * fps))
    end_frame = int(round(clip_end * fps)) if clip_end is not None else None

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index >= start_frame and (end_frame is None or frame_index <= end_frame):
                writer.write(frame)
            if end_frame is not None and frame_index > end_frame:
                break
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if not clip_output.exists():
        raise RuntimeError(f"Failed to create MVBench clip for: {video_path}")
    return clip_output


def _collect_eval_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answered = sum(1 for row in rows if row.get("prediction_letter") is not None)
    correct = sum(1 for row in rows if row.get("correct"))
    parse_methods: dict[str, int] = {}
    task_stats: dict[str, dict[str, int]] = {}

    for row in rows:
        parse_method = str(row.get("parse_method", "unknown"))
        parse_methods[parse_method] = parse_methods.get(parse_method, 0) + 1

        task_name = str(row.get("task_name", "unknown"))
        stats = task_stats.setdefault(task_name, {"total": 0, "answered": 0, "correct": 0})
        stats["total"] += 1
        if row.get("prediction_letter") is not None:
            stats["answered"] += 1
        if row.get("correct"):
            stats["correct"] += 1

    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "answer_rate": (answered / total) if total > 0 else 0.0,
        "parse_methods": dict(sorted(parse_methods.items())),
        "tasks": task_stats,
    }


def _print_task_breakdown(task_stats: dict[str, dict[str, int]]) -> None:
    print("By Task:")
    for task_name, stats in sorted(task_stats.items(), key=lambda item: item[0]):
        total = stats["total"]
        answered = stats["answered"]
        correct = stats["correct"]
        accuracy = (correct / total) if total > 0 else 0.0
        answer_rate = (answered / total) if total > 0 else 0.0
        print(
            f"  {task_name:<24} total={total:<4} answered={answered:<4} "
            f"acc={accuracy:.4f} answer_rate={answer_rate:.4f}"
        )


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig) -> None:
    experiment_path = _resolve_experiment_config_path(config.get("experiment"))
    experiment_config = OmegaConf.load(experiment_path)
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must load as DictConfig: {experiment_path}")

    runtime_config = OmegaConf.merge(experiment_config, config)
    eval_config = runtime_config.get("mvbench")
    if eval_config is None:
        raise ValueError(
            "`mvbench` section must be provided in the eval config. "
            "Use `config/eval.yaml` or pass `--config-name eval`."
        )
    invoke_config = runtime_config.get("invoke")
    if invoke_config is None:
        raise ValueError("`invoke` section must be provided in the experiment config.")

    prompt_file_value = invoke_config.get("prompt_file")
    if not prompt_file_value:
        raise ValueError("`invoke.prompt_file` must be provided in the config.")

    prompt_file = Path(to_absolute_path(str(prompt_file_value))).resolve()
    prompt_template = _load_prompt_template(prompt_file)
    dataset_root, annotations_dir, data_dir = _resolve_dataset_layout(eval_config)
    task_names = list(eval_config.get("tasks") or [])
    annotation_files = _collect_annotation_files(annotations_dir, task_names if task_names else None)
    video_lookup, duplicate_keys = _build_video_lookup(data_dir)
    skip_missing_videos = bool(eval_config.get("skip_missing_videos", False))
    samples, skipped, missing = _load_samples(
        annotation_files=annotation_files,
        video_lookup=video_lookup,
        duplicate_keys=duplicate_keys,
        skip_missing_videos=skip_missing_videos,
    )

    start_index = _resolve_optional_int(eval_config.get("start_index")) or 0
    limit = _resolve_optional_int(eval_config.get("limit"))
    if start_index > 0:
        samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]

    output_dir = _to_abs_path(str(eval_config.get("output_dir")) if eval_config.get("output_dir") else "./eval/result")
    if output_dir is None:
        raise ValueError("Failed to resolve MVBench output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_value = str(eval_config.get("output_file")) if eval_config.get("output_file") else None
    if output_file_value:
        raw_output_path = Path(output_file_value).expanduser()
        if raw_output_path.is_absolute():
            output_path = raw_output_path.resolve()
        elif raw_output_path.parent == Path("."):
            output_path = (output_dir / raw_output_path.name).resolve()
        else:
            output_path = Path(to_absolute_path(str(raw_output_path))).resolve()
    else:
        output_path = (output_dir / "mvbench_eval.jsonl").resolve()

    print("=== MVBench Eval Setup ===")
    print(f"Experiment   : {experiment_path}")
    print(f"Dataset Root : {dataset_root}")
    print(f"Annotations  : {annotations_dir}")
    print(f"Data Dir     : {data_dir}")
    print(f"Prompt File  : {prompt_file}")
    print(f"Output File  : {output_path}")
    print(f"Tasks        : {len(annotation_files)}")
    print(f"Samples      : {len(samples)}")
    print(f"Skipped      : {skipped} (duplicate video names)")
    print(f"Missing Skip : {missing} (missing video paths, enabled={skip_missing_videos})")
    print()

    if invoke_config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(runtime_config, resolve=True).strip())
        print()

    with suppress_model_loading_output(enabled=invoke_config.get("quiet_model_loading", True)):
        vlm = build_vlm(runtime_config)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="mvbench_eval_")
    results: list[dict[str, Any]] = []
    try:
        temp_dir = Path(temp_dir_obj.name)
        dynamic_query_file = temp_dir / "query.txt"
        dynamic_query_enabled, dynamic_query_targets = _configure_dynamic_query_file(vlm, dynamic_query_file)
        if getattr(vlm, "frame_selector", None) is not None or getattr(vlm, "patch_selector", None) is not None:
            if dynamic_query_enabled:
                print(f"Dynamic Query: {dynamic_query_file} -> {', '.join(dynamic_query_targets)}")
            else:
                print("Dynamic Query: no selector exposes a dynamic `query_file` to update.")
            print()

        preload_runtime_resources = getattr(vlm, "preload_runtime_resources", None)
        preloaded = False
        runtime_metric_totals = init_runtime_metric_totals()

        progress_bar = tqdm(samples, desc="MVBench Eval", unit="sample", dynamic_ncols=True, disable=len(samples) == 0)
        for sample in progress_bar:
            clip_path = _extract_video_clip(sample.video_path, sample.start, sample.end, temp_dir)
            prompt = _render_prompt(prompt_template=prompt_template, sample=sample)
            if dynamic_query_enabled:
                dynamic_query_file.write_text(sample.question + "\n", encoding="utf-8")

            if not preloaded and callable(preload_runtime_resources):
                preload_runtime_resources(video_path=str(clip_path), prompt=prompt)
                preloaded = True

            response = vlm.answer(video_path=str(clip_path), prompt=prompt)
            runtime_metrics = extract_runtime_metrics(vlm)
            update_runtime_metric_totals(runtime_metric_totals, runtime_metrics)
            prediction_letter, prediction_index, parse_method = _parse_prediction(response, sample.candidates)
            ground_truth_letter = _option_letter(sample.answer_index)
            is_correct = (
                prediction_index == sample.answer_index
                if prediction_index is not None and sample.answer_index is not None
                else False
            )

            results.append(
                {
                    "task_name": sample.task_name,
                    "sample_index": sample.sample_index,
                    "video_name": sample.video_name,
                    "video_path": str(sample.video_path),
                    "clip_path": str(clip_path),
                    "question": sample.question,
                    "options": sample.candidates,
                    "ground_truth": sample.answer_text,
                    "ground_truth_letter": ground_truth_letter,
                    "prediction": prediction_index,
                    "prediction_letter": prediction_letter,
                    "prediction_option_text": (
                        sample.candidates[prediction_index]
                        if prediction_index is not None and 0 <= prediction_index < len(sample.candidates)
                        else None
                    ),
                    "correct": is_correct,
                    "parse_method": parse_method,
                    "response": response,
                    "start": sample.start,
                    "end": sample.end,
                    "raw_item": sample.raw_item,
                    "dynamic_query_file": str(dynamic_query_file) if dynamic_query_enabled else None,
                    **runtime_metrics,
                }
            )
        progress_bar.close()
    finally:
        temp_dir_obj.cleanup()

    _write_jsonl(output_path, results)
    eval_stats = _collect_eval_stats(results)

    print()
    print("=== MVBench Eval Summary ===")
    print(f"Completed    : {len(results)}")
    print(f"Answered     : {eval_stats['answered']}/{eval_stats['total']}")
    print(f"Correct      : {eval_stats['correct']}/{eval_stats['total']}")
    print(f"Accuracy     : {eval_stats['accuracy']:.4f}")
    print(f"Answer Rate  : {eval_stats['answer_rate']:.4f}")
    runtime_summary = summarize_runtime_metric_totals(runtime_metric_totals)
    for label, formatted_value in format_runtime_summary_lines(runtime_summary):
        print(f"{label:<13}: {formatted_value}")
    print(f"Output File  : {output_path}")
    if eval_stats["parse_methods"]:
        print("Parse Stats  :")
        for parse_method, count in eval_stats["parse_methods"].items():
            print(f"  {parse_method:<16} {count}")
    print()
    _print_task_breakdown(eval_stats["tasks"])


if __name__ == "__main__":
    main()
