#!/usr/bin/env python3
"""Extract one cell from grid TIFF frames and encode it as a video."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop the same cell from frame_*.tiff images and encode an MP4."
    )
    parser.add_argument("frames_dir", type=Path, help="Directory containing frame_*.tiff files.")
    parser.add_argument("output", type=Path, help="Output video path, usually .mp4.")
    parser.add_argument("--grid-cols", type=int, default=4, help="Number of grid columns.")
    parser.add_argument("--grid-rows", type=int, default=4, help="Number of grid rows.")
    parser.add_argument("--cell-x", type=int, default=0, help="Zero-based cell column.")
    parser.add_argument("--cell-y", type=int, default=0, help="Zero-based cell row.")
    parser.add_argument("--cell-size", type=int, default=128, help="Cell width and height.")
    parser.add_argument("--margin", type=int, default=2, help="Outer margin and cell gutter.")
    parser.add_argument("--fps", type=int, default=24, help="Output video frames per second.")
    parser.add_argument(
        "--keep-frames",
        type=Path,
        help="Optional directory for the cropped PNG frame sequence.",
    )
    return parser.parse_args()


def frame_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def crop_box(args: argparse.Namespace) -> tuple[int, int, int, int]:
    if not 0 <= args.cell_x < args.grid_cols:
        raise ValueError("cell-x is outside the grid.")
    if not 0 <= args.cell_y < args.grid_rows:
        raise ValueError("cell-y is outside the grid.")
    if args.cell_size <= 0 or args.margin < 0:
        raise ValueError("cell-size must be positive and margin must be non-negative.")

    step = args.cell_size + args.margin
    left = args.margin + args.cell_x * step
    top = args.margin + args.cell_y * step
    return left, top, left + args.cell_size, top + args.cell_size


def encode_mp4(frame_dir: Path, output: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode MP4 output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-vf",
        "format=yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def write_crops(files: list[Path], frame_dir: Path, box: tuple[int, int, int, int]) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_path in enumerate(files):
        with Image.open(frame_path) as image:
            if box[2] > image.width or box[3] > image.height:
                raise ValueError(
                    f"Crop {box} exceeds image bounds {image.width}x{image.height}."
                )
            crop = image.crop(box).convert("RGB")
            crop.save(frame_dir / f"frame_{index:06d}.png")


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    files = sorted(frames_dir.glob("frame_*.tiff"), key=frame_number)
    if not files:
        raise FileNotFoundError(f"No frame_*.tiff files found in {frames_dir}.")

    box = crop_box(args)
    if args.keep_frames:
        frame_dir = args.keep_frames.expanduser().resolve()
        write_crops(files, frame_dir, box)
        encode_mp4(frame_dir, output, args.fps)
    else:
        with tempfile.TemporaryDirectory(prefix="grid_cell_frames_") as tmp:
            frame_dir = Path(tmp)
            write_crops(files, frame_dir, box)
            encode_mp4(frame_dir, output, args.fps)

    print(f"Wrote {output} from cell ({args.cell_x}, {args.cell_y})")


if __name__ == "__main__":
    main()
