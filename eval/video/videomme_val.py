from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_letter(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    return text[:1] if text else ""


def _accumulate(
    bucket: dict[str, dict[str, int]],
    key: str | None,
    correct: bool,
    missing: bool,
) -> None:
    label = key or "unknown"
    stats = bucket.setdefault(label, {"total": 0, "correct": 0, "missing": 0})
    stats["total"] += 1
    if missing:
        stats["missing"] += 1
    if correct:
        stats["correct"] += 1


def _print_breakdown(title: str, bucket: dict[str, dict[str, int]]) -> None:
    print(title)
    for label, stats in sorted(bucket.items()):
        total = stats["total"]
        correct = stats["correct"]
        missing = stats["missing"]
        acc = (correct / total) * 100 if total else 0.0
        miss_rate = (missing / total) * 100 if total else 0.0
        print(f"  {label:<24} acc={acc:6.2f}%  missing={miss_rate:6.2f}%  ({correct}/{total})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-MME results validator (local accuracy).")
    parser.add_argument(
        "--results_file",
        required=True,
        help="Path to videomme_results.json",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file).expanduser().resolve()

    results_payload = _load_json(results_path)
    if not isinstance(results_payload, list):
        raise ValueError("Results JSON must be a list of video entries.")

    total = 0
    correct = 0
    missing = 0

    by_duration: dict[str, dict[str, int]] = {}
    by_domain: dict[str, dict[str, int]] = {}
    by_sub_category: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}

    for video_item in results_payload:
        questions = video_item.get("questions", [])
        if not isinstance(questions, list):
            continue
        duration = str(video_item.get("duration", "")).strip() or None
        domain = str(video_item.get("domain", "")).strip() or None
        sub_category = str(video_item.get("sub_category", "")).strip() or None
        for question in questions:
            answer = _normalize_letter(question.get("answer"))
            prediction = _normalize_letter(question.get("response"))
            is_missing = not prediction
            is_correct = bool(prediction) and prediction == answer

            total += 1
            if is_missing:
                missing += 1
            if is_correct:
                correct += 1

            _accumulate(by_duration, duration, is_correct, is_missing)
            _accumulate(by_domain, domain, is_correct, is_missing)
            _accumulate(by_sub_category, sub_category, is_correct, is_missing)
            _accumulate(by_task, str(question.get("task_type", "")).strip() or None, is_correct, is_missing)

    accuracy = (correct / total) * 100 if total else 0.0
    missing_rate = (missing / total) * 100 if total else 0.0

    print("=== Video-MME Local Accuracy ===")
    print(f"Total       : {total}")
    print(f"Correct     : {correct}")
    print(f"Accuracy    : {accuracy:.2f}%")
    print(f"Missing     : {missing} ({missing_rate:.2f}%)")
    print()

    _print_breakdown("By Duration", by_duration)
    _print_breakdown("By Domain", by_domain)
    _print_breakdown("By Sub-Category", by_sub_category)
    _print_breakdown("By Task Type", by_task)


if __name__ == "__main__":
    main()
