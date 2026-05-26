from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
INDEX_PATTERN = re.compile(r"\b(?:answer|option|choice)?\s*[:#\-]?\s*([0-9])\b", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b(?:answer|option|choice)?\s*[:#\-]?\s*([A-D])\b", re.IGNORECASE)
TRAILING_COMMA_PATTERN = re.compile(r",(?=\s*[\]}])")


@dataclass(slots=True)
class VideoMMEQuestion:
    video_id: str
    video_name: str
    question_id: str
    duration: str | None
    domain: str | None
    sub_category: str | None
    task_type: str | None
    question: str
    options: list[str]
    answer: str | None
    response: str | None
    video_path: Path
    raw_video_item: dict[str, Any]
    raw_question_item: dict[str, Any]


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _to_abs_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(to_absolute_path(str(path_value))).expanduser().resolve()


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    return resolved if resolved >= 0 else None


def _resolve_experiment_config_path(experiment_value: Any) -> Path:
    if experiment_value is None or not str(experiment_value).strip():
        raise ValueError(
            "`experiment` must be provided. "
            "Example: `python -m eval.videomme experiment=base` or `experiment=trips`."
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


def _normalize_query_text(text: str) -> str:
    return " ".join(str(text).split())


def _normalize_match_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[^0-9a-z\s]+", " ", lowered)
    return " ".join(lowered.split())


def _find_named_path(
    search_root: Path,
    *,
    names: tuple[str, ...],
    expect_dir: bool,
) -> Path | None:
    candidate_roots = [
        search_root,
        search_root / "Video-MME",
        search_root / "video-mme",
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


def _resolve_dataset_layout(eval_config: DictConfig | None) -> tuple[Path, Path, Path | None]:
    dataset_root = _to_abs_path(
        str(eval_config.get("dataset_root")) if eval_config and eval_config.get("dataset_root") else "./Video-MME"
    )
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "Video-MME dataset root could not be found. "
            "Set `videomme.dataset_root` to the dataset directory."
        )

    annotation_file = _to_abs_path(
        str(eval_config.get("annotation_file")) if eval_config and eval_config.get("annotation_file") else None
    )
    if annotation_file is None:
        annotation_file = _to_abs_path(
            str(eval_config.get("template_file")) if eval_config and eval_config.get("template_file") else None
        )
    if annotation_file is None:
        annotation_file = _find_named_path(
            dataset_root,
            names=("test-00000-of-00001.parquet", "output_test_template.json"),
            expect_dir=False,
        )
    if annotation_file is None:
        raise FileNotFoundError(
            "Could not find a Video-MME annotation file. "
            "Set `videomme.annotation_file` to the parquet annotation shard or a compatible JSON file."
        )

    videos_dir = _to_abs_path(
        str(eval_config.get("videos_dir")) if eval_config and eval_config.get("videos_dir") else None
    )
    if videos_dir is None:
        videos_dir = _find_named_path(
            dataset_root,
            names=VIDEO_DIR_CANDIDATES,
            expect_dir=True,
        )
    if videos_dir is None:
        raise FileNotFoundError(
            "Could not find Video-MME video directory. "
            f"Looked for: {VIDEO_DIR_CANDIDATES}"
        )

    video_map_file = _to_abs_path(
        str(eval_config.get("video_map_file")) if eval_config and eval_config.get("video_map_file") else None
    )
    return dataset_root, annotation_file, videos_dir, video_map_file


def _load_json_loose(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        sanitized = TRAILING_COMMA_PATTERN.sub("", text)
        return json.loads(sanitized)


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading Video-MME parquet annotations requires `pandas` with parquet support. "
            "Install `pandas` and `pyarrow` in the runtime."
        ) from exc

    frame = pd.read_parquet(path)
    records = frame.to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    for row in records:
        normalized.append({str(key): value for key, value in row.items()})
    return normalized


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


def _index_videos(videos_dir: Path) -> list[Path]:
    indexed = [
        path.resolve()
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    indexed.sort(key=lambda path: (path.parent.name.casefold(), path.name.casefold()))
    if not indexed:
        raise FileNotFoundError(f"No supported video files were found under: {videos_dir}")
    return indexed


def _load_video_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Video map file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json_loose(path)
        if not isinstance(payload, dict):
            raise ValueError("Video map JSON must be an object of video_id -> filename.")
        return {str(key).strip(): str(value).strip() for key, value in payload.items()}

    if suffix == ".csv":
        mapping: dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                video_id = str(row.get("video_id", "")).strip()
                file_name = str(row.get("file_name", row.get("filename", ""))).strip()
                if video_id and file_name:
                    mapping[video_id] = file_name
        return mapping

    raise ValueError("Unsupported video map file format. Use .json or .csv.")


def _build_video_lookup(indexed_videos: list[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in indexed_videos:
        lookup.setdefault(path.name.casefold(), path)
        lookup.setdefault(path.stem.casefold(), path)
    return lookup


def _resolve_video_path(
    *,
    video_name: str,
    indexed_videos: list[Path],
    video_lookup: dict[str, Path],
    video_map: dict[str, str],
) -> Path:
    normalized_name = video_name.strip()
    direct_candidates = [normalized_name, f"{normalized_name}.mp4"]
    for candidate in direct_candidates:
        candidate_key = candidate.casefold()
        if candidate_key in video_lookup:
            return video_lookup[candidate_key]

    mapped_name = video_map.get(normalized_name)
    if mapped_name:
        mapped_candidates = [mapped_name, Path(mapped_name).name, Path(mapped_name).stem]
        for candidate in mapped_candidates:
            candidate_key = str(candidate).casefold()
            if candidate_key in video_lookup:
                return video_lookup[candidate_key]
        raise FileNotFoundError(
            f"Video map resolved `{normalized_name}` -> `{mapped_name}`, but no matching file was found under the videos dir."
        )

    raise FileNotFoundError(
        f"Could not resolve a video file for video name/id={normalized_name!r}. "
        "Provide `videomme.video_map_file` or ensure the actual mp4 filename matches the annotation `videoID`."
    )


def _parse_option_string(option_text: str) -> list[str]:
    stripped = option_text.strip()
    if not stripped:
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    matches = re.findall(r"([A-D]\.\s*.*?)(?=\s*,\s*[A-D]\.\s*|$)", stripped)
    if matches:
        return [match.strip() for match in matches]
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _extract_options(question_item: dict[str, Any]) -> list[str]:
    options = question_item.get("options")
    if options is not None and not isinstance(options, (list, str)):
        try:
            if hasattr(options, "tolist"):
                options = options.tolist()
            else:
                options = list(options)
        except TypeError:
            pass
    if isinstance(options, str):
        parsed_options = _parse_option_string(options)
        if parsed_options:
            return parsed_options
    if not isinstance(options, list) or not options:
        raise ValueError(f"Question item is missing a non-empty `options` list: {question_item}")
    return [str(option).strip() for option in options]


def _load_samples_from_template(
    *,
    template_file: Path,
    indexed_videos: list[Path],
    video_lookup: dict[str, Path],
    video_map: dict[str, str],
) -> tuple[list[VideoMMEQuestion], list[dict[str, Any]]]:
    payload = _load_json_loose(template_file)
    if not isinstance(payload, list):
        raise TypeError("Video-MME template must be a list of video entries.")

    samples: list[VideoMMEQuestion] = []
    normalized_payload: list[dict[str, Any]] = []

    for video_item in payload:
        if not isinstance(video_item, dict):
            raise TypeError("Each Video-MME video entry must be an object.")
        normalized_video_item = dict(video_item)
        video_id = str(normalized_video_item.get("video_id", "")).strip()
        if not video_id:
            raise ValueError(f"Video-MME video entry is missing `video_id`: {normalized_video_item}")
        video_name = str(normalized_video_item.get("videoID", video_id)).strip()
        video_path = _resolve_video_path(
            video_name=video_name,
            indexed_videos=indexed_videos,
            video_lookup=video_lookup,
            video_map=video_map,
        )

        questions = normalized_video_item.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"Video-MME video entry is missing a non-empty `questions` list: video_id={video_id}")

        normalized_questions: list[dict[str, Any]] = []
        for question_item in questions:
            if not isinstance(question_item, dict):
                raise TypeError(f"Video-MME question entry must be an object: video_id={video_id}")
            normalized_question_item = dict(question_item)
            question_id = str(normalized_question_item.get("question_id", "")).strip()
            question_text = str(normalized_question_item.get("question", "")).strip()
            if not question_id or not question_text:
                raise ValueError(
                    f"Question entry must include `question_id` and `question`: video_id={video_id}, item={normalized_question_item}"
                )
            normalized_questions.append(normalized_question_item)
            samples.append(
                VideoMMEQuestion(
                    video_id=video_id,
                    video_name=video_name,
                    question_id=question_id,
                    duration=str(normalized_video_item.get("duration")).strip()
                    if normalized_video_item.get("duration") is not None
                    else None,
                    domain=str(normalized_video_item.get("domain")).strip()
                    if normalized_video_item.get("domain") is not None
                    else None,
                    sub_category=str(normalized_video_item.get("sub_category")).strip()
                    if normalized_video_item.get("sub_category") is not None
                    else None,
                    task_type=str(normalized_question_item.get("task_type")).strip()
                    if normalized_question_item.get("task_type") is not None
                    else None,
                    question=question_text,
                    options=_extract_options(normalized_question_item),
                    answer=str(normalized_question_item.get("answer")).strip()
                    if normalized_question_item.get("answer") is not None
                    else None,
                    response=str(normalized_question_item.get("response")).strip()
                    if normalized_question_item.get("response") is not None
                    else None,
                    video_path=video_path,
                    raw_video_item=normalized_video_item,
                    raw_question_item=normalized_question_item,
                )
            )

        normalized_video_item["questions"] = normalized_questions
        normalized_video_item["videoID"] = video_name
        normalized_payload.append(normalized_video_item)

    return samples, normalized_payload


def _load_samples_from_parquet(
    *,
    annotation_file: Path,
    indexed_videos: list[Path],
    video_lookup: dict[str, Path],
    video_map: dict[str, str],
) -> tuple[list[VideoMMEQuestion], list[dict[str, Any]]]:
    rows = _load_parquet_rows(annotation_file)
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            raise ValueError(f"Parquet annotation row is missing `video_id`: {row}")
        grouped_rows.setdefault(video_id, []).append(row)

    samples: list[VideoMMEQuestion] = []
    submission_payload: list[dict[str, Any]] = []

    for video_id in sorted(grouped_rows.keys(), key=lambda value: (len(value), value)):
        video_rows = sorted(
            grouped_rows[video_id],
            key=lambda row: str(row.get("question_id", "")).strip(),
        )
        first_row = video_rows[0]
        video_name = str(first_row.get("videoID", "")).strip()
        if not video_name:
            raise ValueError(f"Parquet annotation row is missing `videoID`: {first_row}")
        video_path = _resolve_video_path(
            video_name=video_name,
            indexed_videos=indexed_videos,
            video_lookup=video_lookup,
            video_map=video_map,
        )

        video_payload = {
            "video_id": video_id,
            "duration": _normalize_optional_text(first_row.get("duration")),
            "domain": _normalize_optional_text(first_row.get("domain")),
            "sub_category": _normalize_optional_text(first_row.get("sub_category")),
            "videoID": video_name,
            "url": _normalize_optional_text(first_row.get("url")),
            "questions": [],
        }

        for row in video_rows:
            question_id = str(row.get("question_id", "")).strip()
            question_text = str(row.get("question", "")).strip()
            if not question_id or not question_text:
                raise ValueError(
                    f"Parquet annotation row must include `question_id` and `question`: {row}"
                )

            question_payload = {
                "question_id": question_id,
                "task_type": _normalize_optional_text(row.get("task_type")),
                "question": question_text,
                "options": _extract_options(row),
                "answer": _normalize_optional_text(row.get("answer")),
                "response": None,
            }
            video_payload["questions"].append(question_payload)
            samples.append(
                VideoMMEQuestion(
                    video_id=video_id,
                    video_name=video_name,
                    question_id=question_id,
                    duration=_normalize_optional_text(row.get("duration")),
                    domain=_normalize_optional_text(row.get("domain")),
                    sub_category=_normalize_optional_text(row.get("sub_category")),
                    task_type=_normalize_optional_text(row.get("task_type")),
                    question=question_text,
                    options=question_payload["options"],
                    answer=question_payload["answer"],
                    response=None,
                    video_path=video_path,
                    raw_video_item=video_payload,
                    raw_question_item=question_payload,
                )
            )

        submission_payload.append(video_payload)

    return samples, submission_payload


def _load_samples(
    *,
    annotation_file: Path,
    indexed_videos: list[Path],
    video_lookup: dict[str, Path],
    video_map: dict[str, str],
) -> tuple[list[VideoMMEQuestion], list[dict[str, Any]]]:
    if annotation_file.suffix.lower() == ".parquet":
        return _load_samples_from_parquet(
            annotation_file=annotation_file,
            indexed_videos=indexed_videos,
            video_lookup=video_lookup,
            video_map=video_map,
        )

    return _load_samples_from_template(
        template_file=annotation_file,
        indexed_videos=indexed_videos,
        video_lookup=video_lookup,
        video_map=video_map,
    )


def _build_option_block(options: list[str]) -> str:
    return "\n".join(str(option).strip() for option in options)


def _render_prompt(
    *,
    prompt_template: dict[str, str],
    sample: VideoMMEQuestion,
) -> dict[str, str]:
    option_block = _build_option_block(sample.options)
    query_text = "\n".join(
        [
            "Select the best answer to the following multiple-choice question based on the video.",
            "Respond with only the letter (A, B, C, or D) of the correct option.",
            f"Question: {sample.question}",
            option_block,
            "The best answer is:",
        ]
    )
    format_values = _SafeFormatDict(
        {
            "video_id": sample.video_id,
            "question_id": sample.question_id,
            "question": sample.question,
            "query": query_text,
            "options": option_block,
            "option_block": option_block,
            "candidate_answers": option_block,
            "duration": sample.duration or "",
            "domain": sample.domain or "",
            "sub_category": sample.sub_category or "",
            "task_type": sample.task_type or "",
            **{f"option_{index}": option for index, option in enumerate(sample.options)},
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


def _parse_prediction(response: str, options: list[str]) -> tuple[str, int | None, str]:
    stripped = response.strip()
    if not stripped:
        return "", None, "empty_response"

    for match in LETTER_PATTERN.finditer(stripped.upper()):
        index = ord(match.group(1)) - ord("A")
        if 0 <= index < len(options):
            letter = _option_letter(index)
            return letter or "", index, "letter_regex"

    for match in INDEX_PATTERN.finditer(stripped):
        index = int(match.group(1))
        if 0 <= index < len(options):
            letter = _option_letter(index)
            return letter or "", index, "index_regex"
        if 1 <= index <= len(options):
            adjusted = index - 1
            letter = _option_letter(adjusted)
            return letter or "", adjusted, "one_based_index_regex"

    normalized_response = _normalize_match_text(stripped)
    for index, option in enumerate(options):
        normalized_option = _normalize_match_text(option)
        if normalized_option and (
            normalized_option in normalized_response
            or normalized_response in normalized_option
        ):
            letter = _option_letter(index)
            return letter or "", index, "substring_match"

    return stripped, None, "raw_response"


def _write_json(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_debug_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_submission_stats(
    submission_payload: list[dict[str, Any]],
    debug_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_videos = len(submission_payload)
    total_questions = 0
    filled_responses = 0
    empty_responses = 0
    parse_methods: dict[str, int] = {}

    for video_item in submission_payload:
        questions = video_item.get("questions", [])
        if not isinstance(questions, list):
            continue
        for question in questions:
            total_questions += 1
            response = question.get("response")
            if response is None or not str(response).strip():
                empty_responses += 1
            else:
                filled_responses += 1

    for row in debug_rows:
        parse_method = str(row.get("parse_method", "unknown"))
        parse_methods[parse_method] = parse_methods.get(parse_method, 0) + 1

    return {
        "videos": total_videos,
        "questions": total_questions,
        "filled_responses": filled_responses,
        "empty_responses": empty_responses,
        "parse_methods": dict(sorted(parse_methods.items())),
    }


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig) -> None:
    experiment_path = _resolve_experiment_config_path(config.get("experiment"))
    experiment_config = OmegaConf.load(experiment_path)
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must load as DictConfig: {experiment_path}")

    runtime_config = OmegaConf.merge(experiment_config, config)
    eval_config = runtime_config.get("videomme")
    if eval_config is None:
        raise ValueError(
            "`videomme` section must be provided in the eval config. "
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
    dataset_root, annotation_file, videos_dir, video_map_file = _resolve_dataset_layout(eval_config)
    indexed_videos = _index_videos(videos_dir)
    video_lookup = _build_video_lookup(indexed_videos)
    video_map = _load_video_map(video_map_file)

    samples, submission_payload = _load_samples(
        annotation_file=annotation_file,
        indexed_videos=indexed_videos,
        video_lookup=video_lookup,
        video_map=video_map,
    )

    start_index = _resolve_optional_int(eval_config.get("start_index"))
    limit = _resolve_optional_int(eval_config.get("limit"))
    start_index = start_index or 0
    if start_index > 0:
        samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]

    output_dir = _to_abs_path(
        str(eval_config.get("output_dir")) if eval_config.get("output_dir") else "./outputs/videomme"
    )
    if output_dir is None:
        raise ValueError("Failed to resolve Video-MME output directory.")
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
        output_path = (output_dir / "videomme_results.json").resolve()

    debug_output_path: Path | None = None
    debug_output_value = str(eval_config.get("debug_output_file")) if eval_config.get("debug_output_file") else None
    if debug_output_value:
        raw_debug_path = Path(debug_output_value).expanduser()
        if raw_debug_path.is_absolute():
            debug_output_path = raw_debug_path.resolve()
        elif raw_debug_path.parent == Path("."):
            debug_output_path = (output_dir / raw_debug_path.name).resolve()
        else:
            debug_output_path = Path(to_absolute_path(str(raw_debug_path))).resolve()

    print("=== Video-MME Result Setup ===")
    print(f"Experiment   : {experiment_path}")
    print(f"Dataset Root : {dataset_root}")
    print(f"Annotation   : {annotation_file}")
    print(f"Videos Dir   : {videos_dir}")
    print(f"Prompt File  : {prompt_file}")
    print(f"Result File  : {output_path}")
    if video_map_file is not None:
        print(f"Video Map    : {video_map_file}")
    else:
        print(f"Video Map    : None")
    print(f"Video Files  : {len(indexed_videos)}")
    print(f"Questions    : {len(samples)}")
    if debug_output_path is not None:
        print(f"Debug JSONL  : {debug_output_path}")
    print()

    if invoke_config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(runtime_config, resolve=True).strip())
        print()

    with suppress_model_loading_output(
        enabled=invoke_config.get("quiet_model_loading", True),
    ):
        vlm = build_vlm(runtime_config)

    submission_lookup: dict[str, dict[str, Any]] = {}
    for video_item in submission_payload:
        submission_lookup[str(video_item["video_id"])] = {
            str(question["question_id"]): question
            for question in video_item["questions"]
        }

    temp_dir = tempfile.TemporaryDirectory(prefix="videomme_query_")
    debug_rows: list[dict[str, Any]] = []
    try:
        dynamic_query_file = Path(temp_dir.name) / "query.txt"
        dynamic_query_enabled, dynamic_query_targets = _configure_dynamic_query_file(vlm, dynamic_query_file)
        if getattr(vlm, "frame_selector", None) is not None or getattr(vlm, "patch_selector", None) is not None:
            if dynamic_query_enabled:
                target_list = ", ".join(dynamic_query_targets)
                print(f"Dynamic Query: {dynamic_query_file} -> {target_list}")
            else:
                print("Dynamic Query: no selector exposes a dynamic `query_file` to update.")
            print()
        preload_runtime_resources = getattr(vlm, "preload_runtime_resources", None)
        preloaded = False
        runtime_metric_totals = init_runtime_metric_totals()

        progress_bar = tqdm(
            samples,
            desc="Video-MME Eval",
            unit="question",
            dynamic_ncols=True,
            disable=len(samples) == 0,
        )
        for sample in progress_bar:
            query_text = _normalize_query_text(sample.question)
            if dynamic_query_enabled:
                dynamic_query_file.write_text(query_text + "\n", encoding="utf-8")
            prompt = _render_prompt(
                prompt_template=prompt_template,
                sample=sample,
            )

            if not preloaded and callable(preload_runtime_resources):
                preload_runtime_resources(
                    video_path=str(sample.video_path),
                    prompt=prompt,
                )
                preloaded = True

            response = vlm.answer(
                video_path=str(sample.video_path),
                prompt=prompt,
            )
            runtime_metrics = extract_runtime_metrics(vlm)
            update_runtime_metric_totals(runtime_metric_totals, runtime_metrics)
            prediction_letter, prediction_index, parse_method = _parse_prediction(response, sample.options)
            prediction_option_text = (
                sample.options[prediction_index]
                if prediction_index is not None and 0 <= prediction_index < len(sample.options)
                else None
            )
            target_question = submission_lookup[sample.video_id][sample.question_id]
            target_question["response"] = prediction_letter

            debug_rows.append(
                {
                    "video_id": sample.video_id,
                    "question_id": sample.question_id,
                    "video_path": str(sample.video_path),
                    "question": sample.question,
                    "options": sample.options,
                    "ground_truth": sample.answer,
                    "prediction": prediction_index,
                    "prediction_letter": prediction_letter,
                    "prediction_option_text": prediction_option_text,
                    "parse_method": parse_method,
                    "response": response,
                    "duration": sample.duration,
                    "domain": sample.domain,
                    "sub_category": sample.sub_category,
                    "task_type": sample.task_type,
                    "dynamic_query_file": str(dynamic_query_file) if dynamic_query_enabled else None,
                    **runtime_metrics,
                }
            )
        progress_bar.close()
    finally:
        temp_dir.cleanup()

    _write_json(output_path, submission_payload)
    if debug_output_path is not None:
        _write_debug_jsonl(debug_output_path, debug_rows)
    submission_stats = _collect_submission_stats(submission_payload, debug_rows)

    print()
    print("=== Video-MME Result Summary ===")
    print(f"Completed    : {len(debug_rows)}")
    print(f"Videos       : {submission_stats['videos']}")
    print(f"Questions    : {submission_stats['questions']}")
    print(f"Responses    : {submission_stats['filled_responses']}/{submission_stats['questions']}")
    print(f"Empty Resp   : {submission_stats['empty_responses']}")
    runtime_summary = summarize_runtime_metric_totals(runtime_metric_totals)
    for label, formatted_value in format_runtime_summary_lines(runtime_summary):
        print(f"{label:<13}: {formatted_value}")
    print(f"Result File  : {output_path}")
    if debug_output_path is not None:
        print(f"Debug JSONL  : {debug_output_path}")
    if submission_stats["parse_methods"]:
        print("Parse Stats  :")
        for parse_method, count in submission_stats["parse_methods"].items():
            print(f"  {parse_method:<16} {count}")


if __name__ == "__main__":
    main()
