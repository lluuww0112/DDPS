from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


QUESTION_FILE_CANDIDATES = ("val.csv", "train.csv")


def _make_question_uid(
    q_uid: Any,
    *,
    video_id: Any | None = None,
    fallback: Any | None = None,
) -> str:
    if q_uid not in (None, ""):
        q_uid_text = str(q_uid).strip()
        video_text = str(video_id).strip() if video_id not in (None, "") else ""
        if video_text and not q_uid_text.startswith(f"{video_text}:"):
            return f"{video_text}:{q_uid_text}"
        return q_uid_text
    if fallback is not None:
        return str(fallback)
    raise ValueError("Missing q_uid/qid.")


def _to_abs_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(to_absolute_path(str(path_value))).expanduser().resolve()


def _load_payload(input_path: Path) -> Any:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number} in {input_path}: {exc}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Each JSONL row must be an object, got {type(item).__name__} "
                        f"at line {line_number}."
                    )
                items.append(item)
        return items

    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_prediction(value: Any, *, q_uid: str) -> int:
    if value is None:
        raise ValueError(f"Missing prediction for q_uid={q_uid}.")
    if isinstance(value, bool):
        raise ValueError(f"Boolean prediction is not valid for q_uid={q_uid}.")
    if isinstance(value, int):
        return int(value)

    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty prediction for q_uid={q_uid}.")
    if text.isdigit():
        return int(text)
    raise ValueError(
        f"Prediction must be an integer option index for q_uid={q_uid}, got {value!r}."
    )


def _normalize_submission_payload(raw_payload: Any) -> dict[str, int]:
    if isinstance(raw_payload, dict):
        normalized: dict[str, int] = {}
        for key, value in raw_payload.items():
            video_id = value.get("video_id") if isinstance(value, dict) else None
            q_uid = _make_question_uid(key, video_id=video_id)
            prediction = value.get("prediction", value.get("answer")) if isinstance(value, dict) else value
            normalized[q_uid] = _coerce_prediction(prediction, q_uid=q_uid)
        return normalized

    if isinstance(raw_payload, list):
        normalized: dict[str, int] = {}
        for index, item in enumerate(raw_payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"List payload entries must be objects, got {type(item).__name__} "
                    f"at index {index}."
                )
            q_uid_value = item.get("q_uid") or item.get("qid") or item.get("uid") or item.get("question_id")
            if q_uid_value is None:
                raise ValueError(f"Missing q_uid/qid in list payload at index {index}.")
            q_uid = _make_question_uid(q_uid_value, video_id=item.get("video_id"))
            prediction = item.get("prediction", item.get("answer"))
            normalized[q_uid] = _coerce_prediction(prediction, q_uid=q_uid)
        return normalized

    raise ValueError(
        "Unsupported prediction file format. Expected JSON object, JSON list, or JSONL."
    )


def load_submission_payload(input_path: str | Path) -> dict[str, int]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    raw_payload = _load_payload(path)
    normalized_payload = _normalize_submission_payload(raw_payload)
    if not normalized_payload:
        raise ValueError(f"No predictions found in: {path}")
    return normalized_payload


def _find_named_path(search_root: Path, *, names: tuple[str, ...]) -> Path | None:
    candidate_roots = [
        search_root,
        search_root / "nextqa",
        search_root / "Next-QA",
        search_root / "NExTVideo" / "nextqa",
    ]
    for root in candidate_roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

    for name in names:
        try:
            match = next(search_root.rglob(name))
        except StopIteration:
            match = None
        if match is not None and match.is_file():
            return match.resolve()
    return None


def _load_ground_truth_map(questions_file: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with questions_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            q_uid = _make_question_uid(
                row.get("qid"),
                video_id=row.get("video"),
                fallback=index,
            )
            answer = row.get("answer")
            if answer in (None, ""):
                continue
            labels[q_uid] = int(str(answer).strip())

    if not labels:
        raise ValueError(
            f"No labeled NextQA samples were found in question file: {questions_file}"
        )
    return labels


@hydra.main(version_base=None, config_path="../config", config_name="val")
def main(config: DictConfig) -> None:
    val_config = config.get("nextqa_val")
    if val_config is None:
        raise ValueError("`nextqa_val` section must be provided in config/val.yaml.")

    input_file = _to_abs_path(str(val_config.get("input_file")) if val_config.get("input_file") else None)
    if input_file is None:
        raise ValueError("`nextqa_val.input_file` must be provided.")

    dataset_root = _to_abs_path(
        str(val_config.get("dataset_root")) if val_config.get("dataset_root") else "./nextqa"
    )
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "NextQA dataset root could not be found. Set `nextqa_val.dataset_root`."
        )

    questions_file = _to_abs_path(
        str(val_config.get("questions_file")) if val_config.get("questions_file") else None
    )
    if questions_file is None:
        questions_file = _find_named_path(dataset_root, names=QUESTION_FILE_CANDIDATES)
    if questions_file is None:
        raise FileNotFoundError(
            "Could not find NextQA question file for validation. "
            f"Looked for: {QUESTION_FILE_CANDIDATES}"
        )

    predictions = load_submission_payload(input_file)
    labels = _load_ground_truth_map(questions_file)

    matched = 0
    correct = 0
    missing_predictions: list[str] = []
    extra_predictions = sorted(set(predictions) - set(labels))

    for q_uid, answer in labels.items():
        prediction = predictions.get(q_uid)
        if prediction is None:
            missing_predictions.append(q_uid)
            continue
        matched += 1
        if prediction == answer:
            correct += 1

    accuracy = correct / matched if matched > 0 else 0.0

    print(f"Loaded predictions : {len(predictions)}")
    print(f"Labeled samples    : {len(labels)}")
    print(f"Matched samples    : {matched}")
    print(f"Correct samples    : {correct}")
    print(f"Accuracy           : {accuracy:.4f}" if matched > 0 else "Accuracy           : N/A")
    print(f"Missing preds      : {len(missing_predictions)}")
    print(f"Extra preds        : {len(extra_predictions)}")

    if config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(config, resolve=True).strip())
        print()


if __name__ == "__main__":
    main()
