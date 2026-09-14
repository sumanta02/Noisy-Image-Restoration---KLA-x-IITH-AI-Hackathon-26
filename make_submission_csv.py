from __future__ import annotations

import argparse
import base64
import csv
import os
from io import BytesIO
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create submission.csv from .npy prediction files")
    parser.add_argument("--submission-dir", type=str, default="/kaggle/working/submission")
    parser.add_argument("--output-csv", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--expected-size", type=int, default=256)
    parser.add_argument("--expected-count", type=int, default=-1)
    return parser.parse_args()


def validate_array(arr: np.ndarray, expected_size: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape != (expected_size, expected_size):
        raise ValueError(f"Expected shape ({expected_size}, {expected_size}), got {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Array contains NaN or Inf values")
    return arr


def main() -> None:
    args = parse_args()

    submission_dir = Path(args.submission_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(submission_dir) if f.endswith(".npy")])
    if args.expected_count >= 0 and len(files) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} .npy files, found {len(files)} in {submission_dir}"
        )

    rows = []
    for idx, file_name in enumerate(files, start=1):
        path = submission_dir / file_name
        arr = np.load(path)
        arr = validate_array(arr, expected_size=args.expected_size)

        buffer = BytesIO()
        np.save(buffer, arr)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        rows.append({"id": idx, "npy_base64": encoded})

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "npy_base64"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Submission created with {len(rows)} rows at {output_csv}")


if __name__ == "__main__":
    main()
