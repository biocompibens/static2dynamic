#!/usr/bin/env python3
"""Create media trajectories from time-unpaired ground-truth image folders."""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build videos or GIFs by sampling one image from each time folder. "
            "Expected layout: root/0/*.png, root/1/*.png, ..."
        )
    )
    parser.add_argument("root", type=Path, help="Directory containing timepoint folders.")
    parser.add_argument("output_dir", type=Path, help="Directory where media files are written.")
    parser.add_argument("--count", type=int, default=1, help="Number of media files to create.")
    parser.add_argument("--duration", type=int, default=700, help="Frame duration in ms.")
    parser.add_argument(
        "--total-duration",
        type=float,
        default=10.0,
        help="Total MP4 duration in seconds.",
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output format. MP4 is pauseable and is the default.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--size", type=int, default=0, help="Optional square resize in pixels.")
    parser.add_argument(
        "--order",
        help="Comma-separated timepoint folder order. Unlisted folders are appended after it.",
    )
    return parser.parse_args()


def time_key(path: Path, order: dict[str, int] | None = None) -> tuple[int, int, str]:
    if order and path.name in order:
        return (0, order[path.name], path.name)
    return (1, int(path.name), path.name) if path.name.isdigit() else (2, 10**9, path.name)


def collect_timepoints(root: Path, order: dict[str, int] | None = None) -> list[list[Path]]:
    folders = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: time_key(path, order))
    timepoints: list[list[Path]] = []
    for folder in folders:
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            timepoints.append(images)
    return timepoints


def load_frame(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        frame = image.convert("RGB")
        if size > 0:
            frame = frame.resize((size, size), Image.Resampling.LANCZOS)
        return frame


def write_mp4(
    frames: list[Image.Image], output_path: Path, duration_seconds: float, fps: int = 24
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode MP4 output.")

    with tempfile.TemporaryDirectory(prefix="gt_pairing_frames_") as tmp:
        frame_dir = Path(tmp)
        frame_count = max(1, round(duration_seconds * fps))
        for index in range(frame_count):
            frame = frames[min(len(frames) - 1, (index * len(frames)) // frame_count)]
            frame.save(frame_dir / f"frame_{index:06d}.png")

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be strictly positive.")
    if args.duration <= 0:
        raise ValueError("duration must be strictly positive.")
    if args.total_duration <= 0:
        raise ValueError("total-duration must be strictly positive.")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    order = None
    if args.order:
        order = {name.strip(): index for index, name in enumerate(args.order.split(","))}
    timepoints = collect_timepoints(root, order)
    if not timepoints:
        raise FileNotFoundError(f"No timepoint images found in {root}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    shuffled_timepoints = []
    for images in timepoints:
        shuffled = images[:]
        rng.shuffle(shuffled)
        shuffled_timepoints.append(shuffled)

    for index in range(args.count):
        choices = []
        for images in shuffled_timepoints:
            if args.count <= len(images):
                choices.append(images[index])
            else:
                choices.append(rng.choice(images))
        frames = [load_frame(path, args.size) for path in choices]
        output_path = output_dir / f"ground_truth_random_pairing_{index + 1:02d}.{args.format}"
        if args.format == "mp4":
            write_mp4(frames, output_path, args.total_duration)
        else:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=args.duration,
                loop=1,
            )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
