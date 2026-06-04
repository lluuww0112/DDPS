from __future__ import annotations

import gc
import json
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

from eval.runtime_metrics import (
    extract_runtime_metrics,
    extract_runtime_metrics_from_result,
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
PERIOD_STRIP_PATTERN = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP_PATTERN = re.compile(r"(\d)(\,)(\d)")
PUNCTUATION = tuple(';/[]"{}()=+\\_-><@`,?!')

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
ARTICLES = {"a", "an", "the"}
CONTRACTIONS = dict(
    item.split("=", 1)
    for item in """
    aint=ain't arent=aren't cant=can't couldve=could've couldnt=couldn't
    couldn'tve=couldn't've couldnt've=couldn't've didnt=didn't doesnt=doesn't dont=don't
    hadnt=hadn't hadnt've=hadn't've hadn'tve=hadn't've hasnt=hasn't havent=haven't
    hed=he'd hed've=he'd've he'dve=he'd've hes=he's howd=how'd howll=how'll hows=how's
    Id've=I'd've I'dve=I'd've Im=I'm Ive=I've isnt=isn't itd=it'd itd've=it'd've
    it'dve=it'd've itll=it'll let's=let's maam=ma'am mightnt=mightn't
    mightnt've=mightn't've mightn'tve=mightn't've mightve=might've mustnt=mustn't
    mustve=must've neednt=needn't notve=not've oclock=o'clock oughtnt=oughtn't
    ow's'at='ow's'at 'ows'at='ow's'at 'ow'sat='ow's'at shant=shan't
    shed've=she'd've she'dve=she'd've she's=she's shouldve=should've shouldnt=shouldn't
    shouldnt've=shouldn't've shouldn'tve=shouldn't've somebody'd=somebodyd
    somebodyd've=somebody'd've somebody'dve=somebody'd've somebodyll=somebody'll
    somebodys=somebody's someoned=someone'd someoned've=someone'd've
    someone'dve=someone'd've someonell=someone'll someones=someone's somethingd=something'd
    somethingd've=something'd've something'dve=something'd've somethingll=something'll
    thats=that's thered=there'd thered've=there'd've there'dve=there'd've therere=there're
    theres=there's theyd=they'd theyd've=they'd've they'dve=they'd've theyll=they'll
    theyre=they're theyve=they've twas='twas wasnt=wasn't wed've=we'd've we'dve=we'd've
    weve=we've werent=weren't whatll=what'll whatre=what're whats=what's whatve=what've
    whens=when's whered=where'd wheres=where's whereve=where've whod=who'd
    whod've=who'd've who'dve=who'd've wholl=who'll whos=who's whove=who've whyll=why'll
    whyre=why're whys=why's wont=won't wouldve=would've wouldnt=wouldn't
    wouldnt've=wouldn't've wouldn'tve=wouldn't've yall=y'all yall'll=y'all'll
    y'allll=y'all'll yall'd've=y'all'd've y'alld've=y'all'd've y'all'dve=y'all'd've
    youd=you'd youd've=you'd've you'dve=you'd've youll=you'll youre=you're youve=you've
    """.split()
)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


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


@dataclass(slots=True)
class EvalStats:
    total: int = 0
    scored: int = 0
    score_sum: float = 0.0
    exact_match: int = 0
    answer_types: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def update(self, row: Mapping[str, Any]) -> None:
        self.total += 1
        score = row.get("score")
        if not isinstance(score, (int, float)):
            return

        self.scored += 1
        self.score_sum += float(score)
        if float(score) >= 1.0:
            self.exact_match += 1

        answer_type = str(row.get("answer_type") or "unknown")
        type_stats = self.answer_types.setdefault(
            answer_type,
            {"total": 0, "score_sum": 0.0},
        )
        type_stats["total"] = int(type_stats["total"]) + 1
        type_stats["score_sum"] = float(type_stats["score_sum"]) + float(score)

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "scored": self.scored,
            "accuracy": self.score_sum / self.scored if self.scored else None,
            "exact_match": self.exact_match / self.scored if self.scored else None,
            "answer_types": {
                name: {
                    "total": int(stats["total"]),
                    "accuracy": float(stats["score_sum"]) / int(stats["total"]),
                }
                for name, stats in sorted(self.answer_types.items())
                if int(stats["total"]) > 0
            },
        }


@dataclass(slots=True)
class ExistingOutputState:
    question_ids: set[str] = field(default_factory=set)
    indices: set[int] = field(default_factory=set)
    stats: EvalStats = field(default_factory=EvalStats)


@dataclass(slots=True)
class VQAv2EvalConfig:
    dataset_root: Path
    split: str | None
    require_answers: bool
    output_dir: Path
    output_path: Path
    resume: bool
    start_index: int
    limit: int | None
    batch_size: int
    cleanup_interval: int

    @classmethod
    def from_config(cls, eval_config: DictConfig) -> VQAv2EvalConfig:
        output_dir = _abs_path(eval_config.get("output_dir") or "./eval/result")
        output_file = eval_config.get("output_file") or "vqav2_eval.jsonl"
        dataset_root = cls._dataset_root(eval_config)
        batch_size = _positive_int(eval_config.get("eval_batch_size"), 1)
        return cls(
            dataset_root=dataset_root,
            split=_optional_str(eval_config.get("split")),
            require_answers=bool(eval_config.get("require_answers", False)),
            output_dir=output_dir,
            output_path=_json_path(output_file, output_dir),
            resume=bool(eval_config.get("resume", True)),
            start_index=_optional_int(eval_config.get("start_index")) or 0,
            limit=_optional_int(eval_config.get("limit")),
            batch_size=batch_size,
            cleanup_interval=_optional_int(eval_config.get("empty_cuda_cache_interval")) or 0,
        )

    @staticmethod
    def _dataset_root(eval_config: DictConfig) -> Path:
        root = _abs_path(eval_config.get("dataset_root") or "./data/VQAv2")
        split = _optional_str(eval_config.get("split"))
        if split and root.name != split and (root / split).exists():
            root = root / split
        if not root.exists():
            suffix = f"/{split}" if split else ""
            raise FileNotFoundError(
                "VQAv2 dataset root could not be found. "
                f"Set `vqav2.dataset_root` to the directory created by `vqa_v2.py`{suffix}."
            )
        return root.resolve()


class VQAv2AnswerScorer:
    def clean(self, answer: str) -> str:
        return str(answer).replace("\n", " ").replace("\t", " ").strip()

    def normalize(self, answer: str) -> str:
        return self.process_digit_article(self.process_punctuation(self.clean(answer)))

    def process_punctuation(self, text: str) -> str:
        out_text = text
        has_comma_between_digits = re.search(COMMA_STRIP_PATTERN, text) is not None
        for punct in PUNCTUATION:
            if punct + " " in text or " " + punct in text or has_comma_between_digits:
                out_text = out_text.replace(punct, "")
            else:
                out_text = out_text.replace(punct, " ")
        return PERIOD_STRIP_PATTERN.sub("", out_text, re.UNICODE)

    def process_digit_article(self, text: str) -> str:
        words = [MANUAL_MAP.get(word, word) for word in text.lower().split()]
        words = [word for word in words if word not in ARTICLES]
        return " ".join(CONTRACTIONS.get(word, word) for word in words)

    def score(self, prediction: str, answers: list[str]) -> float | None:
        if not answers:
            return None

        pred = self.normalize(prediction)
        gt_answers = [self.normalize(answer) for answer in answers]

        accuracies = []
        for index, _answer in enumerate(gt_answers):
            other_answers = gt_answers[:index] + gt_answers[index + 1 :]
            matches = sum(1 for answer in other_answers if answer == pred)
            accuracies.append(min(1.0, float(matches) / 3.0))
        return sum(accuracies) / len(accuracies)


class PromptTemplate:
    def __init__(self, path: Path):
        sections = self._load_sections(path)
        if not sections["user"]:
            raise ValueError(f"`[USER]` section is missing in prompt file: {path}")
        self.system = "\n".join(sections["system"]).strip()
        self.user = "\n".join(sections["user"]).strip()

    @staticmethod
    def _load_sections(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        sections = {"system": [], "user": []}
        current: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "[SYSTEM]":
                current = "system"
            elif stripped == "[USER]":
                current = "user"
            elif current is not None:
                sections[current].append(line)
        return sections

    def render(self, sample: VQAv2Sample) -> dict[str, str]:
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


class VQAv2Dataset:
    def __init__(self, root: Path):
        self.root = root
        self.dataset = self._load(root)

    @staticmethod
    def _load(root: Path) -> Any:
        try:
            from datasets import Dataset, DatasetDict, Image as DatasetImage, load_from_disk
        except ImportError as exc:
            raise ImportError(
                "VQAv2 evaluation requires the `datasets` package because "
                "the downloaded split is saved with datasets.save_to_disk()."
            ) from exc

        dataset = load_from_disk(str(root))
        if isinstance(dataset, DatasetDict):
            if len(dataset) != 1:
                raise ValueError("Pass a split directory or set `vqav2.split`.")
            dataset = next(iter(dataset.values()))
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Unsupported VQAv2 dataset object: {type(dataset)!r}")

        for key in IMAGE_KEYS:
            if key in dataset.column_names:
                feature = dataset.features.get(key)
                if isinstance(feature, DatasetImage) and getattr(feature, "decode", True):
                    dataset = dataset.cast_column(key, DatasetImage(decode=False))
                break
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def indices(self, start: int, limit: int | None) -> range:
        first = min(max(start, 0), len(self))
        last = len(self) if limit is None else min(first + limit, len(self))
        return range(first, last)

    def sample(self, index: int) -> VQAv2Sample:
        row = dict(self.dataset[int(index)])
        question = row.get("question")
        if question is None or not str(question).strip():
            raise ValueError(f"VQAv2 row {index} is missing `question`.")

        question_id = _first(row, QUESTION_ID_KEYS, index)
        image_value = _first(row, IMAGE_KEYS)
        if image_value is None:
            raise ValueError(f"VQAv2 row {index} is missing an image column.")

        image_id = _first(row, IMAGE_ID_KEYS)
        multiple_choice_answer = _first(row, MULTIPLE_CHOICE_ANSWER_KEYS)
        return VQAv2Sample(
            index=int(index),
            question_id=str(question_id),
            image_id=str(image_id) if image_id is not None else None,
            question=str(question),
            answers=_answers(_first(row, ANSWER_KEYS)),
            multiple_choice_answer=(
                str(multiple_choice_answer) if multiple_choice_answer is not None else None
            ),
            image_value=image_value,
            raw_item=row,
        )

    def question_ids(self, indices: Iterable[int]) -> Iterable[str]:
        metadata = self._metadata_dataset(QUESTION_ID_KEYS)
        for index in indices:
            row = dict(metadata[int(index)])
            yield str(_first(row, QUESTION_ID_KEYS, int(index)))

    def has_answers(self, indices: Iterable[int], max_probe: int = 32) -> bool:
        if not any(key in self.dataset.column_names for key in ANSWER_KEYS):
            return False

        metadata = self._metadata_dataset(ANSWER_KEYS)
        for offset, index in enumerate(indices):
            if offset >= max_probe:
                break
            row = dict(metadata[int(index)])
            if _answers(_first(row, ANSWER_KEYS)):
                return True
        return False

    def _metadata_dataset(self, keys: tuple[str, ...]) -> Any:
        selected = [key for key in keys if key in self.dataset.column_names]
        if not selected:
            return self.dataset
        select_columns = getattr(self.dataset, "select_columns", None)
        return select_columns(selected) if callable(select_columns) else self.dataset


class ImageMaterializer:
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sample: VQAv2Sample) -> Path:
        value = sample.image_value
        cached_path = self.image_dir / f"{sample.question_id}.jpg"
        if cached_path.exists():
            return cached_path

        if isinstance(value, Image.Image):
            return self._save(value, cached_path)
        if isinstance(value, (str, Path)):
            return self._existing_path(value, sample)
        if isinstance(value, bytes):
            return self._write_bytes(sample, value, ".jpg")
        if isinstance(value, Mapping):
            return self._from_mapping(sample, value, cached_path)
        raise TypeError(
            f"Unsupported VQAv2 image value for question_id={sample.question_id}: "
            f"{type(value)!r}"
        )

    def _from_mapping(
        self,
        sample: VQAv2Sample,
        value: Mapping[str, Any],
        cached_path: Path,
    ) -> Path:
        path_value = value.get("path")
        bytes_value = value.get("bytes")
        if path_value:
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute():
                path = Path(to_absolute_path(str(path))).resolve()
            if path.exists():
                return path
        if isinstance(bytes_value, bytes):
            return self._write_bytes(sample, bytes_value, self._suffix(path_value))
        if bytes_value:
            return self._save(Image.open(BytesIO(bytes_value)), cached_path)
        raise ValueError(
            f"Unsupported VQAv2 image mapping for question_id={sample.question_id}: "
            f"keys={sorted(value.keys())}"
        )

    def _existing_path(self, value: str | Path, sample: VQAv2Sample) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = Path(to_absolute_path(str(path))).resolve()
        if not path.exists():
            raise FileNotFoundError(f"VQAv2 image file not found: {path}")
        return path

    def _write_bytes(self, sample: VQAv2Sample, data: bytes, suffix: str) -> Path:
        path = self.image_dir / f"{sample.question_id}{suffix}"
        if not path.exists():
            path.write_bytes(data)
        return path

    @staticmethod
    def _save(image: Image.Image, path: Path) -> Path:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=95)
        return path

    @staticmethod
    def _suffix(path_value: Any) -> str:
        suffix = Path(str(path_value)).suffix.lower() if path_value else ""
        return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


class JsonlRecorder:
    def __init__(self, path: Path, scorer: VQAv2AnswerScorer, target_qids: set[str] | None):
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
        sample: VQAv2Sample,
        response: str,
        runtime: Mapping[str, Any],
        patch_selection: Any | None = None,
        visual_selection: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "index": sample.index,
            "question_id": sample.question_id,
            "image_id": sample.image_id,
            "question": sample.question,
            "prediction": response,
            "prediction_normalized": self.scorer.normalize(response),
            "multiple_choice_answer": sample.multiple_choice_answer,
            "answers": sample.answers,
            "answer_type": sample.raw_item.get("answer_type"),
            "question_type": sample.raw_item.get("question_type"),
            "score": self.scorer.score(response, sample.answers),
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


class VQAv2Evaluator:
    def __init__(self, hydra_config: DictConfig):
        self.experiment_path = _experiment_config_path(hydra_config.get("experiment"))
        experiment_config = OmegaConf.load(self.experiment_path)
        if not isinstance(experiment_config, DictConfig):
            raise TypeError(f"Experiment config must load as DictConfig: {self.experiment_path}")

        self.runtime_config = OmegaConf.merge(experiment_config, hydra_config)
        self.eval_config = self.runtime_config.get("vqav2")
        self.invoke_config = self.runtime_config.get("invoke")
        if self.eval_config is None:
            raise ValueError("`vqav2` section must be provided in the config.")
        if self.invoke_config is None:
            raise ValueError("`invoke` section must be provided in the experiment config.")

        prompt_file = self.eval_config.get("prompt_file") or self.invoke_config.get(
            "prompt_file"
        )
        if not prompt_file:
            raise ValueError(
                "`vqav2.prompt_file` or `invoke.prompt_file` must be provided in the config."
            )

        self.settings = VQAv2EvalConfig.from_config(self.eval_config)
        _align_maskclip_batch_size(self.runtime_config, self.settings.batch_size)
        self.prompt = PromptTemplate(_abs_path(prompt_file))
        self.dataset = VQAv2Dataset(self.settings.dataset_root)
        self.indices = self.dataset.indices(self.settings.start_index, self.settings.limit)
        self.has_answers = self.dataset.has_answers(self.indices)
        if (
            self.settings.require_answers
            and not self.has_answers
            and not _is_test_split(self.settings.split)
        ):
            raise ValueError(
                "VQAv2 local validation requires public answers, but none were found. "
                "Use the val split or set `vqav2.require_answers=false` for splits without answers."
            )

        target_qids = (
            set(self.dataset.question_ids(self.indices))
            if self.settings.start_index > 0 or self.settings.limit is not None
            else None
        )
        self.recorder = JsonlRecorder(
            self.settings.output_path,
            VQAv2AnswerScorer(),
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
        with tempfile.TemporaryDirectory(prefix="vqav2_eval_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            image_materializer = ImageMaterializer(temp_dir / "images")
            dynamic_query_file = temp_dir / "query.txt"
            dynamic_query_enabled = self._configure_dynamic_query(vlm, dynamic_query_file)
            stats = self._run_samples(
                vlm,
                image_materializer,
                dynamic_query_file,
                dynamic_query_enabled,
                runtime_totals,
            )
        submission_path = self._write_submission_if_needed()
        self._print_summary(stats, runtime_totals, submission_path)

    def _run_samples(
        self,
        vlm: Any,
        image_materializer: ImageMaterializer,
        dynamic_query_file: Path,
        dynamic_query_enabled: bool,
        runtime_totals: dict[str, Any],
    ) -> dict[str, Any]:
        pending: list[VQAv2Sample] = []
        processed_count = 0
        preload = getattr(vlm, "preload_runtime_resources", None)
        preloaded = False

        progress = tqdm(
            self.indices,
            desc="VQAv2 Eval",
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
                    self._flush_batch(vlm, image_materializer, pending, record)
                continue

            prompt = self.prompt.render(sample)
            image_path = image_materializer.path_for(sample)
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
            self._flush_batch(vlm, image_materializer, pending, record)
        return self.recorder.state.stats.summary()

    def _flush_batch(
        self,
        vlm: Any,
        image_materializer: ImageMaterializer,
        samples: list[VQAv2Sample],
        record: Any,
    ) -> None:
        prompts = [self.prompt.render(sample) for sample in samples]
        image_paths = [str(image_materializer.path_for(sample)) for sample in samples]
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

    def _already_done(self, sample: VQAv2Sample) -> bool:
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
        print("=== VQAv2 Eval Setup ===")
        print(f"Experiment   : {self.experiment_path}")
        print(f"Dataset Root : {self.settings.dataset_root}")
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
        if not _is_test_split(self.settings.split):
            return None

        rows_by_qid: dict[str, Mapping[str, Any]] = {}
        for row in self.recorder.iter_result_rows():
            qid = row.get("question_id")
            if qid is None:
                continue
            rows_by_qid[str(qid)] = row

        submission_rows = [
            {
                "question_id": _submission_question_id(row["question_id"]),
                "answer": _submission_answer(row),
            }
            for row in sorted(rows_by_qid.values(), key=_submission_sort_key)
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
        print("=== VQAv2 Eval Complete ===")
        print(f"Output       : {self.settings.output_path}")
        if submission_path is not None:
            print(f"Submission   : {submission_path}")
        print(f"Rows         : {stats['total']}")
        if stats["accuracy"] is None:
            print("Accuracy     : N/A (no public annotations for this split)")
        else:
            print(f"Accuracy     : {float(stats['accuracy']):.4f}")
            print(f"Exact Match  : {float(stats['exact_match']):.4f}")
            if stats["answer_types"]:
                print("By Answer Type:")
                for answer_type, type_stats in stats["answer_types"].items():
                    print(
                        f"  {answer_type:<12} total={type_stats['total']:<6} "
                        f"acc={float(type_stats['accuracy']):.4f}"
                    )
        if self.settings.start_index > 0 or self.settings.limit is not None:
            print("Warning      : subset run; metrics are partial")

        for label, value in format_runtime_summary_lines(summarize_runtime_metric_totals(runtime_totals)):
            print(f"{label:<13}: {value}")


def _abs_path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).expanduser().resolve()


def _optional_str(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    value = int(value)
    if value <= 0:
        raise ValueError(f"Expected a positive integer, got {value}.")
    return value


def _iter_maskclip_patch_selection_configs(value: Any) -> Iterable[DictConfig]:
    if not isinstance(value, DictConfig):
        return

    target = value.get("_target_")
    if target is not None and str(target).endswith("maskclip_patch_selection"):
        yield value

    for child in value.values():
        if isinstance(child, DictConfig):
            yield from _iter_maskclip_patch_selection_configs(child)


def _align_maskclip_batch_size(runtime_config: DictConfig, batch_size: int) -> None:
    seen: set[int] = set()
    candidates = [runtime_config.get("patch_selection")]
    vlm_config = runtime_config.get("vlm")
    if isinstance(vlm_config, DictConfig):
        candidates.append(vlm_config.get("patch_selector"))

    for candidate in candidates:
        for selector_config in _iter_maskclip_patch_selection_configs(candidate):
            selector_id = id(selector_config)
            if selector_id in seen:
                continue
            seen.add(selector_id)
            with open_dict(selector_config):
                selector_config.batch_size = int(batch_size)


def _json_path(value: Any, output_dir: Path) -> Path:
    raw_path = Path(str(value)).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    if raw_path.parent == Path("."):
        return (output_dir / raw_path.name).resolve()
    return _abs_path(raw_path)


def _submission_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_submission.json")


def _is_test_split(split: str | None) -> bool:
    return split is not None and split.strip().lower().startswith("test")


def _submission_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    index = row.get("index")
    qid = str(row.get("question_id", ""))
    if isinstance(index, int):
        return (0, index, qid)
    return (1, 0, qid)


def _submission_question_id(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def _submission_answer(row: Mapping[str, Any]) -> str:
    prediction = row.get("prediction_normalized")
    if prediction is None:
        prediction = row.get("prediction")
    if prediction is None:
        prediction = ""
    return _DEFAULT_SCORER.clean(str(prediction))


def _experiment_config_path(value: Any) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(
            "`experiment` must be provided. "
            "Example: `python -m eval.vqa_v2 experiment=base` or `experiment=DDPS`."
        )

    raw = Path(str(value).strip()).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, CONFIG_DIR / raw]
    if not raw.suffix:
        candidates.extend([Path.cwd() / f"{raw}.yaml", CONFIG_DIR / f"{raw}.yaml"])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find experiment config for `{value}`.")


def _first(row: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(value["answer"])] if value.get("answer") is not None else []
    if isinstance(value, list):
        answers = []
        for item in value:
            answer = item.get("answer") if isinstance(item, Mapping) else item
            if answer is not None:
                answers.append(str(answer))
        return answers
    return [str(value)]


def _batch_item(info: Any, index: int) -> Any:
    if isinstance(info, Mapping):
        items = info.get("items")
        if isinstance(items, list) and index < len(items):
            return items[index]
    return info


def _cleanup_cuda(processed_count: int, interval: int) -> None:
    if interval <= 0 or processed_count <= 0 or processed_count % interval != 0:
        return
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Backward-compatible helpers for small normalization tests/imports.
_DEFAULT_SCORER = VQAv2AnswerScorer()
_normalize_answer = _DEFAULT_SCORER.normalize
_score_vqa_answer = _DEFAULT_SCORER.score


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(config: DictConfig) -> None:
    VQAv2Evaluator(config).run()


if __name__ == "__main__":
    main()
