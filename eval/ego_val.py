from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import requests
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


DEFAULT_URL = "https://validation-server.onrender.com/api/upload/"
DEFAULT_TIMEOUT_SECONDS = 120
VIDEO_UID_FROM_URL_PATTERN = re.compile(r"/([^/?#]+)\.mp4(?:[?#]|$)", re.IGNORECASE)
SKIP_PARSE_METHODS = {"unparsed", "empty_response"}


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


def _extract_video_uid_from_url(url: str) -> str | None:
    match = VIDEO_UID_FROM_URL_PATTERN.search(url.strip())
    if match is None:
        return None
    video_uid = match.group(1).strip()
    return video_uid or None


def _extract_submission_uid(item: dict[str, Any], *, fallback_uid: str) -> str:
    for key in ("video_uid", "video_id", "uid"):
        value = item.get(key)
        if value is not None:
            return str(value)

    uid_metadata = item.get("uid_metadata")
    if isinstance(uid_metadata, dict):
        for key in ("video_uid", "video_id", "uid"):
            value = uid_metadata.get(key)
            if value is not None:
                return str(value)
        for key in ("url", "video_url"):
            value = uid_metadata.get(key)
            if isinstance(value, str):
                video_uid = _extract_video_uid_from_url(value)
                if video_uid is not None:
                    return video_uid

    for key in ("url", "video_url"):
        value = item.get(key)
        if isinstance(value, str):
            video_uid = _extract_video_uid_from_url(value)
            if video_uid is not None:
                return video_uid

    return fallback_uid


def _should_skip_item(item: dict[str, Any]) -> str | None:
    parse_method = item.get("parse_method")
    if isinstance(parse_method, str) and parse_method in SKIP_PARSE_METHODS:
        return f"parse_method={parse_method}"

    if item.get("prediction", item.get("answer")) is None:
        return "missing_prediction"

    return None


def _normalize_submission_payload(raw_payload: Any) -> tuple[dict[str, int], list[str]]:
    if isinstance(raw_payload, dict):
        normalized: dict[str, int] = {}
        skipped: list[str] = []
        for key, value in raw_payload.items():
            q_uid = str(key)
            submission_uid = q_uid
            if isinstance(value, dict):
                skip_reason = _should_skip_item(value)
                if skip_reason is not None:
                    skipped.append(f"{q_uid} ({skip_reason})")
                    continue
                prediction = value.get("prediction", value.get("answer"))
                submission_uid = _extract_submission_uid(value, fallback_uid=q_uid)
            else:
                prediction = value
            try:
                normalized[submission_uid] = _coerce_prediction(prediction, q_uid=submission_uid)
            except ValueError as exc:
                skipped.append(f"{submission_uid} ({exc})")
        return normalized, skipped

    if isinstance(raw_payload, list):
        normalized: dict[str, int] = {}
        skipped: list[str] = []
        for index, item in enumerate(raw_payload, start=1):
            if not isinstance(item, dict):
                skipped.append(
                    f"index={index} (expected object, got {type(item).__name__})"
                )
                continue
            q_uid = item.get("q_uid") or item.get("quid") or item.get("uid")
            if q_uid is None:
                skipped.append(f"index={index} (missing q_uid)")
                continue
            q_uid = str(q_uid)
            skip_reason = _should_skip_item(item)
            if skip_reason is not None:
                skipped.append(f"{q_uid} ({skip_reason})")
                continue
            submission_uid = _extract_submission_uid(item, fallback_uid=q_uid)
            prediction = item.get("prediction", item.get("answer"))
            try:
                normalized[submission_uid] = _coerce_prediction(prediction, q_uid=submission_uid)
            except ValueError as exc:
                skipped.append(f"{submission_uid} ({exc})")
        return normalized, skipped

    raise ValueError(
        "Unsupported prediction file format. Expected JSON object, JSON list, or JSONL."
    )


def load_submission_payload(input_path: str | Path) -> tuple[dict[str, int], list[str]]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    raw_payload = _load_payload(path)
    normalized_payload, skipped_entries = _normalize_submission_payload(raw_payload)
    if not normalized_payload:
        raise ValueError(f"No predictions found in: {path}")
    return normalized_payload, skipped_entries


def send_post_request(
    payload: dict[str, int],
    *,
    url: str = DEFAULT_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> requests.Response:
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    return response


def _to_abs_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(to_absolute_path(str(path_value))).expanduser().resolve()


@hydra.main(version_base=None, config_path="../config", config_name="val")
def main(config: DictConfig) -> None:
    val_config = config.get("egoschema_val")
    if val_config is None:
        raise ValueError("`egoschema_val` section must be provided in config/val.yaml.")

    input_file = _to_abs_path(str(val_config.get("input_file")) if val_config.get("input_file") else None)
    if input_file is None:
        raise ValueError("`egoschema_val.input_file` must be provided.")

    url = str(val_config.get("url") or DEFAULT_URL)
    timeout = int(val_config.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    dry_run = bool(val_config.get("dry_run", False))

    payload, skipped_entries = load_submission_payload(input_file)
    print(f"Loaded predictions : {len(payload)}")
    print(f"Skipped entries    : {len(skipped_entries)}")
    if skipped_entries:
        preview_count = min(len(skipped_entries), 5)
        print("Skipped preview    :")
        for entry in skipped_entries[:preview_count]:
            print(f"  - {entry}")
        remaining = len(skipped_entries) - preview_count
        if remaining > 0:
            print(f"  - ... and {remaining} more")

    if config.get("print_config", False):
        print("=== Resolved Config ===")
        print(OmegaConf.to_yaml(config, resolve=True).strip())
        print()

    if dry_run:
        print("Dry run enabled; upload skipped.")
        return

    response = send_post_request(
        payload,
        url=url,
        timeout=timeout,
    )
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content:\n{response.text}")


if __name__ == "__main__":
    main()
