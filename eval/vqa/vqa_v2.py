from __future__ import annotations

import json
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from eval.runtime_metrics import (
    extract_runtime_metrics,
    format_runtime_summary_lines,
    init_runtime_metric_totals,
    summarize_runtime_metric_totals,
    update_runtime_metric_totals,
)
from model.invoke import build_vlm, suppress_model_loading_output


QUESTION_ID_KEYS = ("question_id", "questionId", "id")
IMAGE_ID_KEYS = ("image_id", "imageId")
ANSWER_KEYS = ("answers", "answer")
MULTIPLE_CHOICE_ANSWER_KEYS = ("multiple_choice_answer", "multipleChoiceAnswer")
IMAGE_KEYS = ("image", "image_path", "image_file", "image_filename")
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
COMMA_BETWEEN_DIGITS_PATTERN = re.compile(r"(?<=\d),(?=\d)")
PUNCTUATION = set(';/[]"{}()=+\\_<>@`,?!-')
PERIOD_STRIP_PATTERN = re.compile(r"(?<!\d)\.(?!\d)")

MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "isnt": "isn't",
    "itll": "it'll",
    "ive": "i've",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}


@dataclass(slots=True)
class VQAv2Sample:
    index: int
    question_id: str
    image_id: str | None
    question: str
    answers: list[str]
    multiple_choice_answer: str | None
    image_value: Any
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


def _resolve_experiment_config_path(experiment_value: Any) -> Path:
    if experiment_value is None or not str(experiment_value).strip():
        raise ValueError(
            "`experiment` must be provided. "
            "Example: `python -m eval.vqa.vqa_v2 experiment=base` or `experiment=DDPS`."
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


def _first_present(item: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        answer = value.get("answer")
        return [str(answer)] if answer is not None else []
    if isinstance(value, list):
        answers: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                answer = item.get("answer")
            else:
                answer = item
            if answer is not None:
                answers.append(str(answer))
        return answers
    return [str(value)]


def _load_dataset(dataset_root: Path) -> Any:
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "VQAv2 evaluation requires the `datasets` package because "
            "`data/download/vqa_v2_download.py` saves with datasets.save_to_disk()."
        ) from exc

    dataset = load_from_disk(str(dataset_root))
    if isinstance(dataset, DatasetDict):
        if len(dataset) != 1:
            available = ", ".join(dataset.keys())
            raise ValueError(
                "VQAv2 dataset root loaded as a DatasetDict. "
                f"Pass a split directory or set `vqav2.split`. Available: {available}"
            )
        dataset = next(iter(dataset.values()))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Unsupported VQAv2 dataset object: {type(dataset)!r}")
    return dataset


def _resolve_dataset_root(eval_config: DictConfig | None) -> Path:
    dataset_root = _to_abs_path(
        str(eval_config.get("dataset_root"))
        if eval_config and eval_config.get("dataset_root")
        else "./data/VQAv2"
    )
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "VQAv2 dataset root could not be found. "
            "Set `vqav2.dataset_root` to the directory created by `vqa_v2_download.py`."
        )
    return dataset_root


def _build_samples(dataset: Any) -> list[VQAv2Sample]:
    samples: list[VQAv2Sample] = []
    for index in range(len(dataset)):
        raw_item = dict(dataset[index])
        question = raw_item.get("question")
        if question is None or not str(question).strip():
            raise ValueError(f"VQAv2 row {index} is missing `question`.")

        question_id = _first_present(raw_item, QUESTION_ID_KEYS)
        if question_id is None:
            question_id = index

        image_value = _first_present(raw_item, IMAGE_KEYS)
        if image_value is None:
            raise ValueError(f"VQAv2 row {index} is missing an image column.")

        multiple_choice_answer = _first_present(raw_item, MULTIPLE_CHOICE_ANSWER_KEYS)
        image_id = _first_present(raw_item, IMAGE_ID_KEYS)
        samples.append(
            VQAv2Sample(
                index=index,
                question_id=str(question_id),
                image_id=str(image_id) if image_id is not None else None,
                question=str(question),
                answers=_normalize_answers(_first_present(raw_item, ANSWER_KEYS)),
                multiple_choice_answer=(
                    str(multiple_choice_answer)
                    if multiple_choice_answer is not None
                    else None
                ),
                image_value=image_value,
                raw_item=raw_item,
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


def _render_prompt(
    *,
    prompt_template: dict[str, str],
    sample: VQAv2Sample,
) -> dict[str, str]:
    format_values = _SafeFormatDict(
        {
            "query": sample.question,
            "question": sample.question,
            "question_id": sample.question_id,
            "image_id": sample.image_id or "",
        }
    )
    return {
        "system": prompt_template["system"].format_map(format_values).strip(),
        "user": prompt_template["user"].format_map(format_values).strip(),
    }


def _configure_dynamic_query_file(vlm: Any, query_file_path: Path) -> tuple[bool, tuple[str, ...]]:
    updated_targets: list[str] = []
    for target_name in ("frame_selector", "patch_selector"):
        target = getattr(vlm, target_name, None)
        if target is None:
            continue
        keywords = getattr(target, "keywords", None)
        if isinstance(keywords, dict) and "query_file" in keywords:
            keywords["query_file"] = str(query_file_path)
            updated_targets.append(target_name)
    return bool(updated_targets), tuple(updated_targets)


def _materialize_image(image_value: Any, image_dir: Path, sample: VQAv2Sample) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / f"{sample.question_id}.jpg"
    if output_path.exists():
        return output_path

    image: Image.Image
    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, (str, Path)):
        path = Path(str(image_value)).expanduser()
        if not path.is_absolute():
            path = Path(to_absolute_path(str(path))).resolve()
        if not path.exists():
            raise FileNotFoundError(f"VQAv2 image file not found: {path}")
        return path
    elif isinstance(image_value, Mapping):
        path_value = image_value.get("path")
        bytes_value = image_value.get("bytes")
        if path_value:
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute():
                path = Path(to_absolute_path(str(path))).resolve()
            if path.exists():
                return path
        if bytes_value:
            image = Image.open(BytesIO(bytes_value))
        else:
            raise ValueError(
                f"Unsupported VQAv2 image mapping for question_id={sample.question_id}: "
                f"keys={sorted(image_value.keys())}"
            )
    elif isinstance(image_value, bytes):
        image = Image.open(BytesIO(image_value))
    else:
        raise TypeError(
            f"Unsupported VQAv2 image value for question_id={sample.question_id}: "
            f"{type(image_value)!r}"
        )

    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(output_path, format="JPEG", quality=95)
    return output_path


def _process_punctuation(text: str) -> str:
    text = COMMA_BETWEEN_DIGITS_PATTERN.sub("", text)
    text = PERIOD_STRIP_PATTERN.sub("", text)
    return "".join(" " if char in PUNCTUATION else char for char in text)


def _normalize_answer(answer: str) -> str:
    text = str(answer).replace("\n", " ").replace("\t", " ").strip().lower()
    text = _process_punctuation(text)
    words = []
    for word in ARTICLE_PATTERN.sub(" ", text).split():
        words.append(MANUAL_MAP.get(word, CONTRACTIONS.get(word, word)))
    return " ".join(words)


def _score_vqa_answer(prediction: str, answers: list[str]) -> float | None:
    if not answers:
        return None

    normalized_prediction = _normalize_answer(prediction)
    normalized_answers = [_normalize_answer(answer) for answer in answers]
    if not normalized_prediction:
        return 0.0

    accuracies: list[float] = []
    for index, _answer in enumerate(normalized_answers):
        other_answers = normalized_answers[:index] + normalized_answers[index + 1 :]
        matching_count = sum(
            1 for answer in other_answers if answer == normalized_prediction
        )
        accuracies.append(min(1.0, matching_count / 3.0))
    return sum(accuracies) / len(accuracies)


def _load_completed_qids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = row.get("question_id")
            if question_id is not None:
                completed.add(str(question_id))
    return completed


def _append_jsonl(output_path: Path, row: Mapping[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_eval_stats(output_path: Path) -> dict[str, Any]:
    total = 0
    scored = 0
    score_sum = 0.0
    exact_match = 0
    answer_types: dict[str, dict[str, float | int]] = {}

    if not output_path.exists():
        return {
            "total": 0,
            "scored": 0,
            "accuracy": None,
            "exact_match": None,
            "answer_types": {},
        }

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            score = row.get("score")
            if isinstance(score, (int, float)):
                scored += 1
                score_sum += float(score)
                if float(score) >= 1.0:
                    exact_match += 1

                answer_type = str(row.get("answer_type") or "unknown")
                type_stats = answer_types.setdefault(
                    answer_type,
                    {"total": 0, "score_sum": 0.0},
                )
                type_stats["total"] = int(type_stats["total"]) + 1
                type_stats["score_sum"] = float(type_stats["score_sum"]) + float(score)

    answer_type_summary = {
        answer_type: {
            "total": int(stats["total"]),
            "accuracy": (
                float(stats["score_sum"]) / int(stats["total"])
                if int(stats["total"]) > 0
                else None
            ),
        }
        for answer_type, stats in sorted(answer_types.items())
    }
    return {
        "total": total,
        "scored": scored,
        "accuracy": (score_sum / scored) if scored > 0 else None,
        "exact_match": (exact_match / scored) if scored > 0 else None,
        "answer_types": answer_type_summary,
    }


def _resolve_output_path(eval_config: DictConfig, output_dir: Path) -> Path:
    output_file_value = str(eval_config.get("output_file")) if eval_config.get("output_file") else None
    if not output_file_value:
        return (output_dir / "vqav2_eval.jsonl").resolve()

    raw_output_path = Path(output_file_value).expanduser()
    if raw_output_path.is_absolute():
        return raw_output_path.resolve()
    if raw_output_path.parent == Path("."):
        return (output_dir / raw_output_path.name).resolve()
    return Path(to_absolute_path(str(raw_output_path))).resolve()


@hydra.main(version_base=None, config_path="../../config", config_name="eval")
def main(config: DictConfig) -> None:
    experiment_path = _resolve_experiment_config_path(config.get("experiment"))
    experiment_config = OmegaConf.load(experiment_path)
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must load as DictConfig: {experiment_path}")

    runtime_config = OmegaConf.merge(experiment_config, config)
    eval_config = runtime_config.get("vqav2")
    if eval_config is None:
        raise ValueError(
            "`vqav2` section must be provided in the eval config. "
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
    dataset_root = _resolve_dataset_root(eval_config)
    dataset = _load_dataset(dataset_root)
    samples = _build_samples(dataset)

    start_index = _resolve_optional_int(eval_config.get("start_index")) or 0
    limit = _resolve_optional_int(eval_config.get("limit"))
    if start_index > 0:
        samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]

    output_dir = _to_abs_path(
        str(eval_config.get("output_dir")) if eval_config.get("output_dir") else "./eval/result"
    )
    if output_dir is None:
        raise ValueError("Failed to resolve VQAv2 output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _resolve_output_path(eval_config, output_dir)

    resume = bool(eval_config.get("resume", True))
    completed_qids = _load_completed_qids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.write_text("", encoding="utf-8")

    print("=== VQAv2 Eval Setup ===")
    print(f"Experiment   : {experiment_path}")
    print(f"Dataset Root : {dataset_root}")
    print(f"Prompt File  : {prompt_file}")
    print(f"Output File  : {output_path}")
    print(f"Samples      : {len(samples)}")
    print(f"Resume       : {resume} (completed={len(completed_qids)})")
    print()

    if invoke_config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(runtime_config, resolve=True).strip())
        print()

    with suppress_model_loading_output(
        enabled=invoke_config.get("quiet_model_loading", True),
    ):
        vlm = build_vlm(runtime_config, inference_task="vqa")

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="vqav2_eval_")
    runtime_metric_totals = init_runtime_metric_totals()
    try:
        temp_dir = Path(temp_dir_obj.name)
        image_dir = temp_dir / "images"
        dynamic_query_file = temp_dir / "query.txt"
        dynamic_query_enabled, dynamic_query_targets = _configure_dynamic_query_file(
            vlm,
            dynamic_query_file,
        )
        if getattr(vlm, "patch_selector", None) is not None:
            if dynamic_query_enabled:
                print(f"Dynamic Query: {dynamic_query_file} -> {', '.join(dynamic_query_targets)}")
            else:
                print("Dynamic Query: no selector exposes a dynamic `query_file` to update.")
            print()

        preload_runtime_resources = getattr(vlm, "preload_runtime_resources", None)
        preloaded = False

        progress_bar = tqdm(
            samples,
            desc="VQAv2 Eval",
            unit="sample",
            dynamic_ncols=True,
            disable=len(samples) == 0,
        )
        for sample in progress_bar:
            if sample.question_id in completed_qids:
                continue

            prompt = _render_prompt(prompt_template=prompt_template, sample=sample)
            image_path = _materialize_image(sample.image_value, image_dir, sample)
            if dynamic_query_enabled:
                dynamic_query_file.write_text(sample.question + "\n", encoding="utf-8")

            if not preloaded and callable(preload_runtime_resources):
                preload_runtime_resources(image_path=str(image_path), prompt=prompt)
                preloaded = True

            response = vlm.answer(
                image_path=str(image_path),
                prompt=prompt,
            )
            runtime_metrics = extract_runtime_metrics(vlm)
            update_runtime_metric_totals(runtime_metric_totals, runtime_metrics)
            score = _score_vqa_answer(response, sample.answers)
            prediction_normalized = _normalize_answer(response)

            row = {
                "index": sample.index,
                "question_id": sample.question_id,
                "image_id": sample.image_id,
                "question": sample.question,
                "prediction": response,
                "prediction_normalized": prediction_normalized,
                "multiple_choice_answer": sample.multiple_choice_answer,
                "answers": sample.answers,
                "answer_type": sample.raw_item.get("answer_type"),
                "question_type": sample.raw_item.get("question_type"),
                "score": score,
                "runtime": runtime_metrics,
                "patch_selection": getattr(vlm, "last_patch_selection_info", {}),
                "visual_selection": getattr(vlm, "last_visual_selection_info", {}),
            }
            _append_jsonl(output_path, row)

            stats = _collect_eval_stats(output_path)
            if stats["accuracy"] is not None:
                progress_bar.set_postfix(
                    acc=f"{float(stats['accuracy']):.4f}",
                    scored=stats["scored"],
                )

        stats = _collect_eval_stats(output_path)

    finally:
        temp_dir_obj.cleanup()

    print("=== VQAv2 Eval Complete ===")
    print(f"Output       : {output_path}")
    print(f"Rows         : {stats['total']}")
    if stats["accuracy"] is not None:
        print(f"Accuracy     : {float(stats['accuracy']):.4f}")
        print(f"Exact Match  : {float(stats['exact_match']):.4f}")
        if stats["answer_types"]:
            print("By Answer Type:")
            for answer_type, type_stats in stats["answer_types"].items():
                accuracy = type_stats["accuracy"]
                formatted = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                print(f"  {answer_type:<12} total={type_stats['total']:<6} acc={formatted}")
    else:
        print("Accuracy     : N/A (this split does not include answers)")

    runtime_summary = summarize_runtime_metric_totals(runtime_metric_totals)
    for label, formatted_value in format_runtime_summary_lines(runtime_summary):
        print(f"{label:<13}: {formatted_value}")


if __name__ == "__main__":
    main()
