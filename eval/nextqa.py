from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
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


QUESTION_FILE_CANDIDATES = ("val.csv", "train.csv")
VIDEO_DIR_CANDIDATES = ("videos", "video", "NExTVideo")
INDEX_PATTERN = re.compile(r"\b(?:answer|option|choice)?\s*[:#\-]?\s*([0-9])\b", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b(?:answer|option|choice)?\s*[:#\-]?\s*([A-E])\b", re.IGNORECASE)


@dataclass(slots=True)
class NextQASample:
    q_uid: str
    video_id: str
    question: str
    options: list[str]
    answer: int | None
    question_type: str | None
    raw_item: dict[str, Any]
    video_path: Path


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


def _resolve_experiment_config_path(experiment_value: Any) -> Path:
    if experiment_value is None or not str(experiment_value).strip():
        raise ValueError(
            "`experiment` must be provided. "
            "Example: `python -m eval.nextqa experiment=base` or `experiment=trips`."
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
        search_root / "nextqa",
        search_root / "Next-QA",
        search_root / "NExTVideo" / "nextqa",
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
        str(eval_config.get("dataset_root")) if eval_config and eval_config.get("dataset_root") else "./nextqa"
    )
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "NextQA dataset root could not be found. "
            "Set `nextqa.dataset_root` to the dataset directory."
        )

    questions_file = _to_abs_path(
        str(eval_config.get("questions_file")) if eval_config and eval_config.get("questions_file") else None
    )
    if questions_file is None:
        questions_file = _find_named_path(
            dataset_root,
            names=QUESTION_FILE_CANDIDATES,
            expect_dir=False,
        )
    if questions_file is None:
        raise FileNotFoundError(
            "Could not find NextQA question file. "
            f"Looked for: {QUESTION_FILE_CANDIDATES}"
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
            "Could not find NextQA video directory. "
            f"Looked for: {VIDEO_DIR_CANDIDATES}"
        )

    return dataset_root, questions_file, videos_dir


def _load_question_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _extract_question_uid(item: dict[str, Any], index: int) -> str:
    q_uid = item.get("qid")
    if q_uid not in (None, ""):
        video_id = str(item.get("video", "")).strip()
        q_uid_text = str(q_uid).strip()
        if video_id:
            return f"{video_id}:{q_uid_text}"
        return q_uid_text
    return str(index)


def _extract_answer_index(item: dict[str, Any]) -> int | None:
    answer = item.get("answer")
    if answer in (None, ""):
        return None
    text = str(answer).strip()
    if text.isdigit():
        return int(text)
    return None


def _extract_options(item: dict[str, Any]) -> list[str]:
    return [str(item[f"a{index}"]).strip() for index in range(5)]


def _resolve_video_path(
    *,
    videos_dir: Path,
    video_id: str,
) -> Path:
    candidates = [
        videos_dir / f"{video_id}.mp4",
        videos_dir / f"{video_id}.avi",
        videos_dir / f"{video_id}.webm",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Some NextQA layouts shard videos into subdirectories such as
    # videos/0000/2440175990.mp4, so fall back to a recursive index by stem.
    indexed_path = _index_videos_by_stem(str(videos_dir.resolve())).get(str(video_id))
    if indexed_path is not None:
        return Path(indexed_path)

    raise FileNotFoundError(
        f"Could not find NextQA video for video_id={video_id} under {videos_dir}"
    )


@lru_cache(maxsize=4)
def _index_videos_by_stem(videos_dir: str) -> dict[str, str]:
    root = Path(videos_dir)
    indexed: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".avi", ".webm"}:
            continue
        indexed.setdefault(path.stem, str(path.resolve()))
    return indexed


def _load_samples(
    *,
    questions_file: Path,
    videos_dir: Path,
) -> list[NextQASample]:
    rows = _load_question_rows(questions_file)
    samples: list[NextQASample] = []
    for index, item in enumerate(rows):
        q_uid = _extract_question_uid(item, index)
        video_id = str(item["video"]).strip()
        samples.append(
            NextQASample(
                q_uid=q_uid,
                video_id=video_id,
                question=str(item["question"]).strip(),
                options=_extract_options(item),
                answer=_extract_answer_index(item),
                question_type=str(item.get("type")).strip() if item.get("type") not in (None, "") else None,
                raw_item=item,
                video_path=_resolve_video_path(videos_dir=videos_dir, video_id=video_id),
            )
        )
    return samples


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


def _build_option_block(options: list[str]) -> str:
    return "\n".join(f"Option {index}: {option}" for index, option in enumerate(options))


def _render_prompt(
    *,
    prompt_template: dict[str, str],
    sample: NextQASample,
) -> dict[str, str]:
    option_block = _build_option_block(sample.options)
    query_text = "\n".join(
        [
            f"Question: {sample.question}",
            "Options:",
            option_block,
            "Answer with only the option number.",
        ]
    )
    format_values = _SafeFormatDict(
        {
            "q_uid": sample.q_uid,
            "qid": sample.q_uid,
            "video_id": sample.video_id,
            "question": sample.question,
            "query": query_text,
            "options": option_block,
            "option_block": option_block,
            "candidate_answers": option_block,
            "num_options": len(sample.options),
            "question_type": sample.question_type or "",
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


def _parse_prediction_index(
    response: str,
    options: list[str],
) -> tuple[int | None, str]:
    stripped = response.strip()
    if not stripped:
        return None, "empty_response"

    for match in INDEX_PATTERN.finditer(stripped):
        index = int(match.group(1))
        if 0 <= index < len(options):
            return index, "index_regex"

    for match in LETTER_PATTERN.finditer(stripped.upper()):
        index = ord(match.group(1)) - ord("A")
        if 0 <= index < len(options):
            return index, "letter_regex"

    normalized_response = _normalize_match_text(stripped)
    best_index: int | None = None
    best_score = -1.0
    second_best = -1.0

    for index, option in enumerate(options):
        normalized_option = _normalize_match_text(option)
        if not normalized_option:
            continue
        if normalized_option in normalized_response or normalized_response in normalized_option:
            return index, "substring_match"

        score = SequenceMatcher(None, normalized_response, normalized_option).ratio()
        if score > best_score:
            second_best = best_score
            best_score = score
            best_index = index
        elif score > second_best:
            second_best = score

    if best_index is not None and best_score >= 0.72 and (best_score - second_best) >= 0.05:
        return best_index, "fuzzy_match"
    return None, "unparsed"


def _load_completed_qids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            q_uid = payload.get("q_uid")
            if q_uid is not None:
                completed.add(str(q_uid))
    return completed


def _write_result(output_path: Path, payload: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_accuracy(correct: int, labeled_count: int) -> str:
    if labeled_count <= 0:
        return "N/A"
    return f"{(correct / labeled_count):.4f}"


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig) -> None:
    experiment_path = _resolve_experiment_config_path(config.get("experiment"))
    experiment_config = OmegaConf.load(experiment_path)
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must load as DictConfig: {experiment_path}")

    runtime_config = OmegaConf.merge(experiment_config, config)
    eval_config = runtime_config.get("nextqa")
    if eval_config is None:
        raise ValueError(
            "`nextqa` section must be provided in the eval config. "
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
    dataset_root, questions_file, videos_dir = _resolve_dataset_layout(eval_config)
    samples = _load_samples(
        questions_file=questions_file,
        videos_dir=videos_dir,
    )

    start_index = _resolve_optional_int(eval_config.get("start_index"))
    limit = _resolve_optional_int(eval_config.get("limit"))
    start_index = start_index or 0
    if start_index > 0:
        samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]

    output_dir = _to_abs_path(
        str(eval_config.get("output_dir")) if eval_config.get("output_dir") else "./outputs/nextqa"
    )
    if output_dir is None:
        raise ValueError("Failed to resolve NextQA output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_value = str(eval_config.get("output_file")) if eval_config.get("output_file") else None
    if not output_file_value:
        output_path = output_dir / "predictions.jsonl"
    else:
        raw_output_path = Path(output_file_value).expanduser()
        if raw_output_path.is_absolute():
            output_path = raw_output_path.resolve()
        elif raw_output_path.parent == Path("."):
            output_path = (output_dir / raw_output_path.name).resolve()
        else:
            output_path = Path(to_absolute_path(str(raw_output_path))).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume = bool(eval_config.get("resume", True))
    progress_interval = int(eval_config.get("progress_interval", 10))
    progress_interval = max(progress_interval, 1)
    completed_qids = _load_completed_qids(output_path) if resume else set()
    skipped_existing = 0
    if resume and completed_qids:
        pending_samples = [
            sample for sample in samples
            if sample.q_uid not in completed_qids
        ]
        skipped_existing = len(samples) - len(pending_samples)
    else:
        pending_samples = samples

    print("=== NextQA Evaluation Setup ===")
    print(f"Experiment   : {experiment_path}")
    print(f"Dataset Root : {dataset_root}")
    print(f"Questions    : {questions_file}")
    print(f"Videos Dir   : {videos_dir}")
    print(f"Prompt File  : {prompt_file}")
    print(f"Output File  : {output_path}")
    print(f"Samples      : {len(samples)}")
    print(f"Pending      : {len(pending_samples)}")
    if skipped_existing > 0:
        print(f"Resume Skip  : {skipped_existing}")
    print()

    if invoke_config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(runtime_config, resolve=True).strip())
        print()

    with suppress_model_loading_output(
        enabled=invoke_config.get("quiet_model_loading", True),
    ):
        vlm = build_vlm(runtime_config)

    temp_dir = tempfile.TemporaryDirectory(prefix="nextqa_query_")
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

        attempted = 0
        completed = 0
        parse_failures = 0
        labeled_count = 0
        correct = 0
        runtime_metric_totals = init_runtime_metric_totals()

        progress_bar = tqdm(
            pending_samples,
            desc="NextQA Eval",
            unit="sample",
            dynamic_ncols=True,
            disable=len(pending_samples) == 0,
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

            attempted += 1
            response = vlm.answer(
                video_path=str(sample.video_path),
                prompt=prompt,
            )
            runtime_metrics = extract_runtime_metrics(vlm)
            update_runtime_metric_totals(runtime_metric_totals, runtime_metrics)
            prediction, parse_method = _parse_prediction_index(response, sample.options)
            if prediction is None:
                parse_failures += 1

            is_correct: bool | None = None
            if sample.answer is not None:
                labeled_count += 1
                is_correct = prediction == sample.answer
                if is_correct:
                    correct += 1

            result = {
                "q_uid": sample.q_uid,
                "video_id": sample.video_id,
                "video_path": str(sample.video_path),
                "query_text": query_text if dynamic_query_enabled else None,
                "question": sample.question,
                "options": sample.options,
                "ground_truth": sample.answer,
                "prediction": prediction,
                "prediction_text": sample.options[prediction] if prediction is not None else None,
                "correct": is_correct,
                "parse_method": parse_method,
                "response": response,
                "question_type": sample.question_type,
                "raw_item": sample.raw_item,
                "dynamic_query_file": str(dynamic_query_file) if dynamic_query_enabled else None,
                **runtime_metrics,
            }
            _write_result(output_path, result)
            completed += 1

            if attempted == 1 or attempted % progress_interval == 0:
                progress_bar.set_postfix(
                    completed=completed,
                    parse_fail=parse_failures,
                    acc=_format_accuracy(correct, labeled_count),
                    refresh=False,
                )
        progress_bar.close()

        print()
        print("=== NextQA Evaluation Summary ===")
        print(f"Attempted    : {attempted}")
        print(f"Completed    : {completed}")
        print(f"Skipped      : {skipped_existing}")
        print(f"Parse Fail   : {parse_failures}")
        if labeled_count > 0:
            print(f"Labeled      : {labeled_count}")
            print(f"Accuracy     : {correct / labeled_count:.4f}")
        else:
            print("Accuracy     : N/A (no ground-truth labels found)")
        runtime_summary = summarize_runtime_metric_totals(runtime_metric_totals)
        for label, formatted_value in format_runtime_summary_lines(runtime_summary):
            print(f"{label:<13}: {formatted_value}")
    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
