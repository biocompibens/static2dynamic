#!/usr/bin/env python3
"""Create a display copy of a video with a fixed duration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retiming helper for website media.")
    parser.add_argument("input", type=Path, help="Input video.")
    parser.add_argument("output", type=Path, help="Output MP4.")
    parser.add_argument("--duration", type=float, default=10.0, help="Target duration in seconds.")
    parser.add_argument("--fps", type=int, default=24, help="Output frame rate.")
    return parser.parse_args()


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be strictly positive.")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required.")

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_duration = probe_duration(source)
    if source_duration <= 0:
        raise ValueError(f"Could not determine a positive duration for {source}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    scale = args.duration / source_duration
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"setpts={scale:.12f}*PTS,fps={args.fps},pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-t",
        str(args.duration),
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
