from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

DATASET_ID: Final = "lmms-lab/VQAv2"
SPLIT_ALIASES: Final = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SPLIT_ALIASES:
        valid_names = ", ".join(sorted(SPLIT_ALIASES))
        raise ValueError(f"Unknown split `{split}`. Use one of: {valid_names}")
    return SPLIT_ALIASES[normalized]


def download_vqav2_split(split: str, output_dir: str | Path) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Downloading lmms-lab/VQAv2 requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc

    resolved_split = normalize_split(split)
    save_path = Path(output_dir).expanduser().resolve()

    if save_path.exists() and any(save_path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {save_path}. "
            "Choose an empty directory or remove the existing contents first."
        )

    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} split `{resolved_split}`...")
    dataset = load_dataset(DATASET_ID, split=resolved_split)

    print(f"Saving dataset to {save_path}...")
    dataset.save_to_disk(str(save_path))

    print(f"Done. rows={len(dataset)} path={save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a full lmms-lab/VQAv2 split and save it with datasets.save_to_disk()."
    )
    parser.add_argument(
        "positional_split",
        nargs="?",
        help="Dataset split to download. Examples: train, validation, val, test.",
    )
    parser.add_argument(
        "positional_output_dir",
        nargs="?",
        help="Directory where the downloaded split will be saved.",
    )
    parser.add_argument(
        "--split",
        dest="split",
        help="Dataset split to download. Examples: train, validation, val, test.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        help="Directory where the downloaded split will be saved.",
    )
    args = parser.parse_args()

    split = args.split or args.positional_split
    output_dir = args.output_dir or args.positional_output_dir
    if split is None or output_dir is None:
        parser.error("both split and output_dir are required")

    download_vqav2_split(split, output_dir)


if __name__ == "__main__":
    main()
