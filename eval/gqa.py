from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from eval.runtime_metrics import (
    extract_runtime_metrics,
    extract_runtime_metrics_from_result,
    format_runtime_summary_lines,
    init_runtime_metric_totals,
    summarize_runtime_metric_totals,
    update_runtime_metric_totals,
)
from eval.vqa_v2 import (
    PromptTemplate,
    _SafeFormatDict,
    _align_maskclip_batch_size,
    _batch_item,
    _cleanup_cuda,
    _experiment_config_path,
    _json_path,
    _optional_int,
    _optional_str,
    _positive_int,
)
from model.invoke import build_vlm, suppress_model_loading_output


QUESTION_ID_KEYS = ("question_id", "questionId", "id")
IMAGE_ID_KEYS = ("image_id", "imageId", "image")
IMAGE_PATH_KEYS = ("image_path", "imagePath", "image_file", "imageFile", "image_filename")
ANSWER_KEYS = ("answer", "answers")
QUESTION_FILE_PATTERNS = (
    "{split}_balanced_questions.json",
    "{split}_all_questions.json",
    "{split}_questions.json",
    "{split}.json",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "testdev": "testdev",
    "test-dev": "testdev",
    "test_dev": "testdev",
    "test": "test",
}


@dataclass(slots=True)
class GQASample:
    index: int
    question_id: str
    image_id: str | None
    question: str
    answer: str | None
    image_value: Any
    raw_item: dict[str, Any]


@dataclass(slots=True)
class EvalStats:
    total: int = 0
    balanced_total: int = 0
    scored: int = 0
    correct: int = 0

    def update(self, row: Mapping[str, Any]) -> None:
        self.total += 1
        score = row.get("score")
        if not isinstance(score, (int, float)):
            return
        if not _question_is_balanced(row):
            return

        self.balanced_total += 1
        self.scored += 1
        if float(score) >= 1.0:
            self.correct += 1

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "balanced_total": self.balanced_total,
            "scored": self.scored,
            "accuracy": self.correct / self.scored if self.scored else None,
            "correct": self.correct,
        }


@dataclass(slots=True)
class ExistingOutputState:
    question_ids: set[str] = field(default_factory=set)
    indices: set[int] = field(default_factory=set)
    stats: EvalStats = field(default_factory=EvalStats)


@dataclass(slots=True)
class GQAEvalConfig:
    dataset_root: Path
    split: str | None
    question_file: Path
    require_answers: bool
    output_dir: Path
    output_path: Path
    resume: bool
    start_index: int
    limit: int | None
    batch_size: int
    cleanup_interval: int

    @classmethod
    def from_config(cls, eval_config: DictConfig) -> GQAEvalConfig:
        output_dir = _abs_path(eval_config.get("output_dir") or "./eval/result/GQA")
        output_file = eval_config.get("output_file") or "gqa_eval.jsonl"
        dataset_root = _abs_path(eval_config.get("dataset_root") or "./data/GQA")
        split = _normalize_split(_optional_str(eval_config.get("split")) or "testdev")
        if not dataset_root.exists():
            raise FileNotFoundError(
                "GQA dataset root could not be found. "
                "Place the GQA images/questions under `./data/GQA` or set `gqa.dataset_root`."
            )

        question_file = _resolve_question_file(dataset_root, split)
        return cls(
            dataset_root=dataset_root.resolve(),
            split=split,
            question_file=question_file,
            require_answers=bool(eval_config.get("require_answers", False)),
            output_dir=output_dir,
            output_path=_json_path(output_file, output_dir),
            resume=bool(eval_config.get("resume", True)),
            start_index=_optional_int(eval_config.get("start_index")) or 0,
            limit=_optional_int(eval_config.get("limit")),
            batch_size=_positive_int(eval_config.get("eval_batch_size"), 1),
            cleanup_interval=_optional_int(eval_config.get("empty_cuda_cache_interval")) or 0,
        )


class GQAAnswerScorer:
    def clean(self, answer: str) -> str:
        return str(answer).replace("\n", " ").replace("\t", " ").strip()

    def normalize(self, answer: Any) -> str:
        return _officialize_gqa_answer(answer)

    def score(self, prediction: str, answer: str | None) -> float | None:
        if answer is None or not str(answer).strip():
            return None
        return float(self.normalize(prediction) == self.normalize(answer))


class GQAPromptTemplate(PromptTemplate):
    def render(self, sample: GQASample) -> dict[str, str]:
        values = _SafeFormatDict(
            query=sample.question,
            question=sample.question,
            question_id=sample.question_id,
            image_id=sample.image_id or "",
        )
        return {
            "system": self.system.format_map(values).strip(),
            "user": self.user.format_map(values).strip(),
            "query": sample.question,
        }


class GQADataset:
    def __init__(self, root: Path, question_file: Path):
        self.root = root
        self.question_file = question_file
        self.records = list(self._load_questions(question_file))
        self._image_index: dict[str, Path] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def indices(self, start: int, limit: int | None) -> range:
        first = min(max(start, 0), len(self))
        last = len(self) if limit is None else min(first + limit, len(self))
        return range(first, last)

    def sample(self, index: int) -> GQASample:
        question_id, item = self.records[int(index)]
        question = item.get("question")
        if question is None or not str(question).strip():
            raise ValueError(f"GQA row {index} is missing `question`.")

        image_id = _first(item, IMAGE_ID_KEYS)
        image_value = _first(item, IMAGE_PATH_KEYS)
        if image_value is None:
            image_value = image_id
        if image_value is None:
            raise ValueError(f"GQA row {index} is missing `imageId` or an image path.")

        answer = _single_answer(_first(item, ANSWER_KEYS))
        return GQASample(
            index=int(index),
            question_id=str(_first(item, QUESTION_ID_KEYS, question_id)),
            image_id=str(image_id) if image_id is not None else None,
            question=str(question),
            answer=answer,
            image_value=image_value,
            raw_item=dict(item),
        )

    def question_ids(self, indices: Iterable[int]) -> Iterable[str]:
        for index in indices:
            question_id, item = self.records[int(index)]
            yield str(_first(item, QUESTION_ID_KEYS, question_id))

    def has_answers(self, indices: Iterable[int], max_probe: int = 32) -> bool:
        for offset, index in enumerate(indices):
            if offset >= max_probe:
                break
            _question_id, item = self.records[int(index)]
            if _single_answer(_first(item, ANSWER_KEYS)):
                return True
        return False

    def image_path_for(self, sample: GQASample) -> Path:
        value = sample.image_value
        if isinstance(value, (str, Path)):
            path = self._resolve_existing_path(value)
            if path is not None:
                return path

            image_id = _strip_image_suffix(str(value))
            path = self._resolve_image_id(image_id)
            if path is not None:
                return path

        if sample.image_id:
            path = self._resolve_image_id(_strip_image_suffix(sample.image_id))
            if path is not None:
                return path

        raise FileNotFoundError(
            f"GQA image file not found for question_id={sample.question_id}, "
            f"image_id={sample.image_id}, image_value={sample.image_value!r}."
        )

    @staticmethod
    def _load_questions(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, Mapping):
                        raise TypeError(f"GQA JSONL row {index} is not an object.")
                    question_id = _first(item, QUESTION_ID_KEYS, index)
                    yield str(question_id), dict(item)
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and isinstance(data.get("questions"), list):
            data = data["questions"]

        if isinstance(data, Mapping):
            for question_id, item in data.items():
                if not isinstance(item, Mapping):
                    raise TypeError(f"GQA question `{question_id}` is not an object.")
                yield str(question_id), dict(item)
            return

        if isinstance(data, list):
            for index, item in enumerate(data):
                if not isinstance(item, Mapping):
                    raise TypeError(f"GQA question row {index} is not an object.")
                question_id = _first(item, QUESTION_ID_KEYS, index)
                yield str(question_id), dict(item)
            return

        raise TypeError(f"Unsupported GQA question file format: {path}")

    def _resolve_existing_path(self, value: str | Path) -> Path | None:
        raw = Path(str(value)).expanduser()
        candidates = [raw] if raw.is_absolute() else [self.root / raw, Path(to_absolute_path(str(raw)))]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        return None

    def _resolve_image_id(self, image_id: str) -> Path | None:
        if not image_id:
            return None

        direct_dirs = [
            self.root / "images",
            self.root / "image",
            self.root / "imgs",
            self.root,
        ]
        for directory in direct_dirs:
            for suffix in IMAGE_SUFFIXES:
                candidate = directory / f"{image_id}{suffix}"
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()

        index = self._get_image_index()
        return index.get(image_id)

    def _get_image_index(self) -> dict[str, Path]:
        if self._image_index is not None:
            return self._image_index

        index: dict[str, Path] = {}
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(path.stem, path.resolve())
        self._image_index = index
        return index


class JsonlRecorder:
    def __init__(self, path: Path, scorer: GQAAnswerScorer, target_qids: set[str] | None):
        self.path = path
        self.scorer = scorer
        self.target_qids = target_qids
        self.state = self._load()

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.state = ExistingOutputState()

    def row_for(
        self,
        vlm: Any,
        sample: GQASample,
        response: str,
        runtime: Mapping[str, Any],
        patch_selection: Any | None = None,
        visual_selection: Any | None = None,
    ) -> dict[str, Any]:
        prediction_normalized = self.scorer.normalize(response)
        answer_normalized = (
            self.scorer.normalize(sample.answer) if sample.answer is not None else None
        )
        return {
            "index": sample.index,
            "question_id": sample.question_id,
            "image_id": sample.image_id,
            "question": sample.question,
            "prediction": response,
            "prediction_normalized": prediction_normalized,
            "answer": sample.answer,
            "answer_normalized": answer_normalized,
            "score": self.scorer.score(response, sample.answer),
            "is_balanced": sample.raw_item.get("isBalanced"),
            "runtime": dict(runtime),
            "patch_selection": patch_selection
            if patch_selection is not None
            else getattr(vlm, "last_patch_selection_info", {}),
            "visual_selection": visual_selection
            if visual_selection is not None
            else getattr(vlm, "last_visual_selection_info", {}),
        }

    def append(self, row: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.state.question_ids.add(str(row["question_id"]))
        self.state.indices.add(int(row["index"]))
        self.state.stats.update(row)

    def runtime_totals(self, vlm: Any) -> dict[str, float | int]:
        totals = init_runtime_metric_totals()
        for row in self._iter_result_rows():
            update_runtime_metric_totals(
                totals,
                extract_runtime_metrics_from_result(row, vlm),
            )
        return totals

    def iter_result_rows(self) -> Iterable[Mapping[str, Any]]:
        yield from self._iter_result_rows()

    def _load(self) -> ExistingOutputState:
        state = ExistingOutputState()
        for row in self._iter_result_rows():
            qid = row.get("question_id")
            if qid is not None:
                state.question_ids.add(str(qid))
            if isinstance(row.get("index"), int):
                state.indices.add(row["index"])
            state.stats.update(row)
        return state

    def _iter_result_rows(self) -> Iterable[Mapping[str, Any]]:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("question_id")
                qid = str(qid) if qid is not None else None
                if self.target_qids is not None and (qid is None or qid not in self.target_qids):
                    continue
                yield row


class GQAEvaluator:
    def __init__(self, hydra_config: DictConfig):
        self.experiment_path = _experiment_config_path(hydra_config.get("experiment"))
        experiment_config = OmegaConf.load(self.experiment_path)
        if not isinstance(experiment_config, DictConfig):
            raise TypeError(f"Experiment config must load as DictConfig: {self.experiment_path}")

        self.runtime_config = OmegaConf.merge(experiment_config, hydra_config)
        self.eval_config = self.runtime_config.get("gqa")
        self.invoke_config = self.runtime_config.get("invoke")
        if self.eval_config is None:
            raise ValueError("`gqa` section must be provided in the config.")
        if self.invoke_config is None:
            raise ValueError("`invoke` section must be provided in the experiment config.")

        prompt_file = self.eval_config.get("prompt_file") or self.invoke_config.get(
            "prompt_file"
        )
        if not prompt_file:
            raise ValueError(
                "`gqa.prompt_file` or `invoke.prompt_file` must be provided in the config."
            )

        self.settings = GQAEvalConfig.from_config(self.eval_config)
        _align_maskclip_batch_size(self.runtime_config, self.settings.batch_size)
        self.prompt = GQAPromptTemplate(_abs_path(prompt_file))
        self.dataset = GQADataset(self.settings.dataset_root, self.settings.question_file)
        self.indices = self.dataset.indices(self.settings.start_index, self.settings.limit)
        self.has_answers = self.dataset.has_answers(self.indices)
        if self.settings.require_answers and not self.has_answers:
            raise ValueError(
                "GQA local validation requires answers, but none were found. "
                "Use a split with public answers or set `gqa.require_answers=false`."
            )

        target_qids = (
            set(self.dataset.question_ids(self.indices))
            if self.settings.start_index > 0 or self.settings.limit is not None
            else None
        )
        self.recorder = JsonlRecorder(
            self.settings.output_path,
            GQAAnswerScorer(),
            target_qids,
        )
        if not self.settings.resume:
            self.recorder.reset()

    def run(self) -> None:
        self._print_setup()
        if self.invoke_config.get("print_config", False):
            print("=== Resolved Config ===")
            print(OmegaConf.to_yaml(self.runtime_config, resolve=True).strip())
            print()

        vlm = self._load_vlm()
        runtime_totals = self.recorder.runtime_totals(vlm)
        with tempfile.TemporaryDirectory(prefix="gqa_eval_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            dynamic_query_file = temp_dir / "query.txt"
            dynamic_query_enabled = self._configure_dynamic_query(vlm, dynamic_query_file)
            stats = self._run_samples(
                vlm,
                dynamic_query_file,
                dynamic_query_enabled,
                runtime_totals,
            )
        submission_path = self._write_submission_if_needed()
        self._print_summary(stats, runtime_totals, submission_path)

    def _run_samples(
        self,
        vlm: Any,
        dynamic_query_file: Path,
        dynamic_query_enabled: bool,
        runtime_totals: dict[str, Any],
    ) -> dict[str, Any]:
        pending: list[GQASample] = []
        processed_count = 0
        preload = getattr(vlm, "preload_runtime_resources", None)
        preloaded = False

        progress = tqdm(
            self.indices,
            desc="GQA Eval",
            unit="sample",
            total=len(self.indices),
            dynamic_ncols=True,
            disable=len(self.indices) == 0,
        )

        def record(row: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
            nonlocal processed_count
            update_runtime_metric_totals(runtime_totals, runtime)
            self.recorder.append(row)
            processed_count += 1
            _cleanup_cuda(processed_count, self.settings.cleanup_interval)
            stats = self.recorder.state.stats.summary()
            if stats["accuracy"] is not None:
                progress.set_postfix(acc=f"{float(stats['accuracy']):.4f}", scored=stats["scored"])

        for sample_index in progress:
            sample = self.dataset.sample(int(sample_index))
            if self._already_done(sample):
                continue

            if self.settings.batch_size > 1:
                pending.append(sample)
                if len(pending) >= self.settings.batch_size:
                    self._flush_batch(vlm, pending, record)
                continue

            prompt = self.prompt.render(sample)
            image_path = self.dataset.image_path_for(sample)
            if dynamic_query_enabled:
                dynamic_query_file.write_text(sample.question + "\n", encoding="utf-8")
            if not preloaded and callable(preload):
                preload(image_path=str(image_path), prompt=prompt)
                preloaded = True

            response = vlm.answer(image_path=str(image_path), prompt=prompt)
            runtime = extract_runtime_metrics(vlm)
            row = self.recorder.row_for(vlm, sample, response, runtime)
            record(row, runtime)

        if pending:
            self._flush_batch(vlm, pending, record)
        return self.recorder.state.stats.summary()

    def _flush_batch(
        self,
        vlm: Any,
        samples: list[GQASample],
        record: Any,
    ) -> None:
        prompts = [self.prompt.render(sample) for sample in samples]
        image_paths = [str(self.dataset.image_path_for(sample)) for sample in samples]
        responses = vlm.answer_batch(
            prompt=prompts,
            image_paths=image_paths,
            batch_size=self.settings.batch_size,
        )
        if len(responses) != len(samples):
            raise RuntimeError(
                "VLM batch response count does not match request count: "
                f"responses={len(responses)}, samples={len(samples)}."
            )

        timing_info = getattr(vlm, "last_timing_info", {})
        patch_info = getattr(vlm, "last_patch_selection_info", {})
        visual_info = getattr(vlm, "last_visual_selection_info", {})
        for index, (sample, response) in enumerate(zip(samples, responses)):
            item_timing = _batch_item(timing_info, index)
            if not isinstance(item_timing, Mapping):
                item_timing = timing_info
            item_patch = _batch_item(patch_info, index)
            item_visual = _batch_item(visual_info, index)
            runtime = extract_runtime_metrics(
                vlm,
                timing_info=item_timing if isinstance(item_timing, Mapping) else None,
                patch_info=item_patch if isinstance(item_patch, Mapping) else None,
                prefer_patch_sequence_lengths=True,
            )
            row = self.recorder.row_for(
                vlm,
                sample,
                response,
                runtime,
                patch_selection=item_patch,
                visual_selection=item_visual,
            )
            record(row, runtime)
        samples.clear()

    def _already_done(self, sample: GQASample) -> bool:
        state = self.recorder.state
        if sample.index in state.indices:
            return True
        if sample.question_id in state.question_ids:
            state.indices.add(sample.index)
            return True
        return False

    def _load_vlm(self) -> Any:
        print("Loading VLM  : starting", flush=True)
        with suppress_model_loading_output(
            enabled=self.invoke_config.get("quiet_model_loading", True),
        ):
            vlm = build_vlm(self.runtime_config, inference_task="vqa")
        print("Loading VLM  : done", flush=True)
        return vlm

    def _configure_dynamic_query(self, vlm: Any, query_file: Path) -> bool:
        targets = []
        for name in ("frame_selector", "patch_selector"):
            target = getattr(vlm, name, None)
            keywords = getattr(target, "keywords", None)
            if isinstance(keywords, dict) and "query_file" in keywords:
                keywords["query_file"] = str(query_file)
                targets.append(name)
        if getattr(vlm, "patch_selector", None) is not None:
            message = (
                f"Dynamic Query: {query_file} -> {', '.join(targets)}"
                if targets
                else "Dynamic Query: no selector exposes a dynamic `query_file` to update."
            )
            print(message)
            print()
        return bool(targets)

    def _print_setup(self) -> None:
        state = self.recorder.state
        print("=== GQA Eval Setup ===")
        print(f"Experiment   : {self.experiment_path}")
        print(f"Dataset Root : {self.settings.dataset_root}")
        print(f"Question File: {self.settings.question_file}")
        if self.settings.split:
            print(f"Dataset Split: {self.settings.split}")
        print(f"Output File  : {self.settings.output_path}")
        print(f"Samples      : {len(self.indices)}")
        print(f"Local Scores : {'yes' if self.has_answers else 'no'}")
        print(f"Resume       : {self.settings.resume} (completed={len(state.question_ids)})")
        if self.settings.batch_size > 1:
            print(f"Eval Batch   : {self.settings.batch_size}")
        if self.settings.cleanup_interval > 0:
            print(f"CUDA Cleanup : every {self.settings.cleanup_interval} processed sample(s)")
        print()

    def _write_submission_if_needed(self) -> Path | None:
        if self.settings.split is None or not self.settings.split.startswith("test"):
            return None

        rows_by_qid: dict[str, Mapping[str, Any]] = {}
        for row in self.recorder.iter_result_rows():
            qid = row.get("question_id")
            if qid is None:
                continue
            rows_by_qid[str(qid)] = row

        submission_rows = [
            {"questionId": qid, "prediction": _submission_answer(row)}
            for qid, row in sorted(rows_by_qid.items(), key=lambda item: item[0])
        ]

        submission_path = _submission_path(self.settings.output_path)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission_path.write_text(
            json.dumps(submission_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return submission_path

    def _print_summary(
        self,
        stats: dict[str, Any],
        runtime_totals: dict[str, Any],
        submission_path: Path | None,
    ) -> None:
        print("=== GQA Eval Complete ===")
        print(f"Output       : {self.settings.output_path}")
        if submission_path is not None:
            print(f"Submission   : {submission_path}")
        print(f"Rows         : {stats['total']}")
        print(f"Balanced Rows: {stats['balanced_total']}")
        if stats["accuracy"] is None:
            print("Accuracy     : N/A (no public annotations for this split)")
        else:
            print(f"Accuracy     : {float(stats['accuracy']) * 100:.2f}%")
            print(f"Correct      : {stats['correct']}/{stats['scored']}")
        if self.settings.start_index > 0 or self.settings.limit is not None:
            print("Warning      : subset run; metrics are partial")

        for label, value in format_runtime_summary_lines(summarize_runtime_metric_totals(runtime_totals)):
            print(f"{label:<13}: {value}")


def _officialize_gqa_answer(answer: Any) -> str:
    return str(answer).replace("\n", " ").replace("\t", " ").strip().lower().rstrip(".").strip()


def _question_is_balanced(row: Mapping[str, Any]) -> bool:
    value = row.get("is_balanced", row.get("isBalanced", True))
    return bool(value)


def _abs_path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).expanduser().resolve()


def _first(row: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _single_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        answer = value.get("answer")
        return str(answer) if answer is not None else None
    if isinstance(value, list):
        for item in value:
            answer = item.get("answer") if isinstance(item, Mapping) else item
            if answer is not None:
                return str(answer)
        return None
    return str(value)


def _normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return SPLIT_ALIASES.get(normalized, normalized)


def _resolve_question_file(root: Path, split: str | None) -> Path:
    if split is None:
        matches = sorted(root.rglob("*questions*.json")) + sorted(root.rglob("*questions*.jsonl"))
        if len(matches) == 1:
            return matches[0].resolve()
        if not matches:
            raise FileNotFoundError(f"No GQA question JSON/JSONL file found under {root}.")
        raise ValueError(
            "Multiple GQA question files were found. Set `gqa.split` to choose one."
        )

    names = [pattern.format(split=split) for pattern in QUESTION_FILE_PATTERNS]
    search_dirs = [root, root / "eval_script", root / "questions1.2", root / "questions", root / "questions1.2" / "questions"]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0].resolve()

    expected = ", ".join(names)
    raise FileNotFoundError(
        f"No GQA question file for split `{split}` was found under {root}. "
        f"Expected one of: {expected}."
    )


def _strip_image_suffix(value: str) -> str:
    path = Path(value)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return path.stem
    return str(value).strip()


def _submission_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_submission.json")


def _submission_answer(row: Mapping[str, Any]) -> str:
    prediction = row.get("prediction_normalized")
    if prediction is None:
        prediction = row.get("prediction")
    if prediction is None:
        prediction = ""
    return _DEFAULT_SCORER.clean(str(prediction))


_DEFAULT_SCORER = GQAAnswerScorer()
_normalize_answer = _DEFAULT_SCORER.normalize
_score_gqa_answer = _DEFAULT_SCORER.score


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig) -> None:
    GQAEvaluator(config).run()


if __name__ == "__main__":
    main()
