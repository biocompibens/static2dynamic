#!/usr/bin/env python3
"""Subsample marker traces inside standalone Plotly HTML exports."""

from __future__ import annotations

import argparse
import base64
import json
import random
import struct
from pathlib import Path
from typing import Any


DTYPE_FORMATS = {
    "f4": "f",
    "f8": "d",
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lighter Plotly HTML by subsampling marker traces."
    )
    parser.add_argument("input", type=Path, help="Standalone Plotly HTML file.")
    parser.add_argument("output", type=Path, help="Output HTML file.")
    parser.add_argument("--max-points", type=int, default=10_000, help="Max points per marker trace.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed.")
    return parser.parse_args()


def find_matching(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Could not find matching JSON delimiter.")


def extract_data_json(text: str) -> tuple[int, int, list[dict[str, Any]]]:
    call_start = text.rfind("Plotly.newPlot(")
    if call_start < 0:
        raise ValueError("Could not find Plotly.newPlot call.")
    data_start = text.find("[", call_start)
    data_end = find_matching(text, data_start, "[", "]")
    return data_start, data_end + 1, json.loads(text[data_start : data_end + 1])


def typed_array_len(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    dtype = value.get("dtype")
    bdata = value.get("bdata")
    if dtype not in DTYPE_FORMATS or not isinstance(bdata, str):
        return None
    item_size = struct.calcsize("<" + DTYPE_FORMATS[dtype])
    return len(base64.b64decode(bdata)) // item_size


def subsample_typed_array(value: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    dtype = value["dtype"]
    fmt = "<" + DTYPE_FORMATS[dtype]
    item_size = struct.calcsize(fmt)
    raw = base64.b64decode(value["bdata"])
    packed = bytearray()
    for index in indices:
        offset = index * item_size
        packed.extend(raw[offset : offset + item_size])
    sampled = dict(value)
    sampled["bdata"] = base64.b64encode(bytes(packed)).decode("ascii")
    return sampled


def subsample_value(value: Any, n_points: int, indices: list[int]) -> Any:
    typed_len = typed_array_len(value)
    if typed_len == n_points:
        return subsample_typed_array(value, indices)
    if isinstance(value, list) and len(value) == n_points:
        return [value[index] for index in indices]
    if isinstance(value, list):
        return [subsample_value(item, n_points, indices) for item in value]
    if isinstance(value, dict):
        return {key: subsample_value(item, n_points, indices) for key, item in value.items()}
    return value


def trace_point_count(trace: dict[str, Any]) -> int | None:
    lengths = [typed_array_len(trace.get(axis)) for axis in ("x", "y", "z")]
    if all(length is not None for length in lengths) and len(set(lengths)) == 1:
        return lengths[0]
    list_lengths = [
        len(trace.get(axis))
        for axis in ("x", "y", "z")
        if isinstance(trace.get(axis), list)
    ]
    if len(list_lengths) == 3 and len(set(list_lengths)) == 1:
        return list_lengths[0]
    return None


def subsample_trace(trace: dict[str, Any], max_points: int, rng: random.Random) -> tuple[dict[str, Any], int, int]:
    if "markers" not in str(trace.get("mode", "")):
        return trace, 0, 0
    n_points = trace_point_count(trace)
    if n_points is None or n_points <= max_points:
        return trace, n_points or 0, n_points or 0
    indices = sorted(rng.sample(range(n_points), max_points))
    return subsample_value(trace, n_points, indices), n_points, max_points


def main() -> None:
    args = parse_args()
    if args.max_points <= 0:
        raise ValueError("max-points must be strictly positive.")

    text = args.input.read_text()
    data_start, data_end, data = extract_data_json(text)
    rng = random.Random(args.seed)

    total_before = 0
    total_after = 0
    new_data = []
    for trace in data:
        new_trace, before, after = subsample_trace(trace, args.max_points, rng)
        total_before += before
        total_after += after
        new_data.append(new_trace)

    replacement = json.dumps(new_data, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text[:data_start] + replacement + text[data_end:])
    print(f"{args.input}: marker points {total_before} -> {total_after}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
