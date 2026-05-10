#!/usr/bin/env python3
"""Build website ground-truth image series from ordered image folders or videos."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PNG image series. With --input-root, one image is selected from each "
            "ordered subfolder. Without --input-root, positional inputs can be images or videos."
        )
    )
    parser.add_argument("output", type=Path, help="Output PNG.")
    parser.add_argument("inputs", nargs="*", type=Path, help="Input images/videos when not using --input-root.")
    parser.add_argument("--input-root", type=Path, help="Directory whose subfolders are ordered cross-sections.")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based image index per subfolder.")
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--cols", type=int, default=0, help="Columns. Defaults to one row.")
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--trim-background", action="store_true")
    parser.add_argument("--trim-threshold", type=int, default=8)
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    parts: list[object] = []
    token = ""
    for char in path.name:
        if char.isdigit() or char == ".":
            token += char
        else:
            if token:
                try:
                    parts.append(float(token))
                except ValueError:
                    parts.append(token)
                token = ""
            parts.append(char)
    if token:
        try:
            parts.append(float(token))
        except ValueError:
            parts.append(token)
    return parts


def image_files(path: Path) -> list[Path]:
    return sorted(
        [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def select_from_root(root: Path, sample_index: int) -> list[Path]:
    folders = sorted([item for item in root.iterdir() if item.is_dir()], key=natural_key)
    if not folders:
        raise FileNotFoundError(f"No subfolders found in {root}.")
    selected = []
    for folder in folders:
        files = image_files(folder)
        if sample_index >= len(files):
            raise ValueError(f"{folder} only has {len(files)} image(s); cannot use sample {sample_index}.")
        selected.append(files[sample_index])
    return selected


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_video_frame(video: Path, timestamp: float, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.4f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def trim_background(image: Image.Image, threshold: int) -> Image.Image:
    background = image.getpixel((0, 0))
    background_is_light = min(background) > 230
    pixels = image.load()
    left, top = image.width, image.height
    right, bottom = -1, -1

    for y in range(image.height):
        for x in range(image.width):
            pixel = pixels[x, y]
            is_foreground = (
                min(pixel) < 250 - threshold
                if background_is_light
                else max(abs(pixel[channel] - background[channel]) for channel in range(3)) > threshold
            )
            if is_foreground:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    if right < left or bottom < top:
        return image

    pad = 2
    return image.crop(
        (
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad + 1),
            min(image.height, bottom + pad + 1),
        )
    )


def load_thumb(path: Path, size: int, trim: bool, trim_threshold: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if trim:
            image = trim_background(image, trim_threshold)
        scale = min(size / image.width, size / image.height)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        thumb = Image.new("RGB", (size, size), "#ffffff")
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        thumb.paste(image, (x, y))
        return thumb


def expanded_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_root:
        if args.inputs:
            raise ValueError("Do not pass positional inputs together with --input-root.")
        return select_from_root(args.input_root.expanduser().resolve(), args.sample_index)
    if not args.inputs:
        raise ValueError("Pass input images/videos or --input-root.")
    return [item.expanduser().resolve() for item in args.inputs]


def main() -> None:
    args = parse_args()
    if args.frames_per_video <= 0 or args.thumb_size <= 0 or args.gap < 0:
        raise ValueError("frames-per-video and thumb-size must be positive; gap must be non-negative.")

    inputs = expanded_inputs(args)
    frame_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="ground_truth_series_") as tmp:
        tmp_dir = Path(tmp)
        for input_index, path in enumerate(inputs):
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTS:
                frame_paths.append(path)
            elif suffix in VIDEO_EXTS:
                if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
                    raise RuntimeError("ffmpeg and ffprobe are required for video inputs.")
                total = duration(path)
                for frame_index in range(args.frames_per_video):
                    timestamp = total * (frame_index + 0.5) / args.frames_per_video
                    frame_path = tmp_dir / f"video_{input_index:02d}_frame_{frame_index:02d}.png"
                    extract_video_frame(path, timestamp, frame_path)
                    frame_paths.append(frame_path)
            else:
                raise ValueError(f"Unsupported input type: {path}")

        thumbs = [
            load_thumb(path, args.thumb_size, args.trim_background, args.trim_threshold)
            for path in frame_paths
        ]

    cols = args.cols if args.cols > 0 else len(thumbs)
    rows = math.ceil(len(thumbs) / cols)
    width = cols * args.thumb_size + (cols - 1) * args.gap
    height = rows * args.thumb_size + (rows - 1) * args.gap
    sheet = Image.new("RGB", (width, height), "#ffffff")

    for index, thumb in enumerate(thumbs):
        col = index % cols
        row = index // cols
        x = col * (args.thumb_size + args.gap)
        y = row * (args.thumb_size + args.gap)
        sheet.paste(thumb, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote {args.output} with {len(thumbs)} images")


if __name__ == "__main__":
    main()
