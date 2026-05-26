from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yt_dlp

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video into the repository data directory.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory where the video will be saved (default: {DATA_DIR})",
    )
    return parser.parse_args()


def download_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    has_ffmpeg = shutil.which("ffmpeg") is not None

    # Prefer H.264/MP4 streams so OpenCV can decode them reliably in Colab.
    preferred_mp4 = (
        "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
        "bestvideo[ext=mp4][vcodec!*=av01]+bestaudio[ext=m4a]/"
        "best[ext=mp4][vcodec!*=av01]/"
        "best[ext=mp4]/best"
    )
    ydl_opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "format": preferred_mp4 if has_ffmpeg else "best[ext=mp4][vcodec!*=av01]/best[ext=mp4]/best",
    }
    if has_ffmpeg:
        ydl_opts["merge_output_format"] = "mp4"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))

        requested_downloads = info.get("requested_downloads") or []
        if requested_downloads:
            merged_path = requested_downloads[0].get("filepath")
            if merged_path:
                return Path(merged_path)

        if downloaded_path.suffix != ".mp4":
            possible_merged = downloaded_path.with_suffix(".mp4")
            if possible_merged.exists():
                return possible_merged

        return downloaded_path


def main() -> None:
    args = parse_args()
    saved_path = download_video(args.url, args.output_dir.resolve())
    print(f"Saved video to: {saved_path}")


if __name__ == "__main__":
    main()
