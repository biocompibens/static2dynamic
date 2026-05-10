#!/usr/bin/env python3
"""Crop one cell from every generated grid frame and encode it as MP4."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop the same cell from frame_*.tiff/png grid images and encode a video."
    )
    parser.add_argument("frames_dir", type=Path, help="Directory containing frame_*.tiff/png files.")
    parser.add_argument("output", type=Path, help="Output MP4 path.")
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--cell-x", type=int, default=0, help="Zero-based grid column.")
    parser.add_argument("--cell-y", type=int, default=0, help="Zero-based grid row.")
    parser.add_argument("--cell-size", type=int, help="Cell width/height. Inferred by default.")
    parser.add_argument("--margin", type=int, default=2, help="Outer margin and cell gutter.")
    parser.add_argument("--duration", type=float, default=10.0, help="Output duration in seconds.")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate.")
    parser.add_argument("--keep-frames", type=Path, help="Optional directory for cropped PNGs.")
    return parser.parse_args()


def frame_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def frame_files(frames_dir: Path) -> list[Path]:
    files = sorted(frames_dir.glob("frame_*.tiff"), key=frame_number)
    if not files:
        files = sorted(frames_dir.glob("frame_*.tif"), key=frame_number)
    if not files:
        files = sorted(frames_dir.glob("frame_*.png"), key=frame_number)
    if not files:
        raise FileNotFoundError(f"No frame_*.tiff/tif/png files found in {frames_dir}.")
    return files


def infer_cell_size(image_size: int, grid_count: int, margin: int) -> int:
    usable = image_size - margin * (grid_count + 1)
    if usable <= 0 or usable % grid_count != 0:
        raise ValueError(
            f"Cannot infer cell size from image size {image_size}, grid {grid_count}, margin {margin}."
        )
    return usable // grid_count


def crop_box(args: argparse.Namespace, sample: Image.Image) -> tuple[int, int, int, int]:
    if not 0 <= args.cell_x < args.grid_cols or not 0 <= args.cell_y < args.grid_rows:
        raise ValueError("cell-x/cell-y is outside the grid.")
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("duration and fps must be positive.")
    if args.margin < 0:
        raise ValueError("margin must be non-negative.")

    cell_w = args.cell_size or infer_cell_size(sample.width, args.grid_cols, args.margin)
    cell_h = args.cell_size or infer_cell_size(sample.height, args.grid_rows, args.margin)
    if cell_w != cell_h:
        raise ValueError(f"Inferred non-square cell {cell_w}x{cell_h}. Pass --cell-size explicitly.")

    step = cell_w + args.margin
    left = args.margin + args.cell_x * step
    top = args.margin + args.cell_y * step
    box = (left, top, left + cell_w, top + cell_h)
    if box[2] > sample.width or box[3] > sample.height:
        raise ValueError(f"Crop {box} exceeds image bounds {sample.width}x{sample.height}.")
    return box


def write_crops(files: list[Path], frame_dir: Path, box: tuple[int, int, int, int]) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_path in enumerate(files):
        with Image.open(frame_path) as image:
            image.crop(box).convert("RGB").save(frame_dir / f"frame_{index:06d}.png")


def encode_mp4(frame_dir: Path, frame_count: int, output: Path, duration: float, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required.")
    input_fps = frame_count / duration
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{input_fps:.12f}",
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-an",
        "-vf",
        f"fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-t",
        f"{duration:.6f}",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    files = frame_files(frames_dir)
    with Image.open(files[0]) as sample:
        box = crop_box(args, sample)

    if args.keep_frames:
        frame_dir = args.keep_frames.expanduser().resolve()
        write_crops(files, frame_dir, box)
        encode_mp4(frame_dir, len(files), output, args.duration, args.fps)
    else:
        with tempfile.TemporaryDirectory(prefix="grid_cell_frames_") as tmp:
            frame_dir = Path(tmp)
            write_crops(files, frame_dir, box)
            encode_mp4(frame_dir, len(files), output, args.duration, args.fps)

    print(f"Wrote {output} from cell ({args.cell_x}, {args.cell_y}) over {len(files)} frames")


if __name__ == "__main__":
    main()
