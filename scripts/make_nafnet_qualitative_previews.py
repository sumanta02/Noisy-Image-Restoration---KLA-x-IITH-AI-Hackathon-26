from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create qualitative preview images for NAFNet predictions")
    parser.add_argument("--input-dir", type=str, default="data/test/NoisyLR")
    parser.add_argument("--pred-dir", type=str, default="outputs/nafnet_h2/test_predictions_best_infer")
    parser.add_argument("--output-dir", type=str, default="outputs/nafnet_h2/qualitative_best_infer")
    parser.add_argument("--sample-ids", type=str, default="0,25,75,150,250,399")
    parser.add_argument("--cmap", type=str, default="gray")
    return parser


def parse_sample_ids(raw: str, max_len: int) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= max_len:
            raise ValueError(f"Sample id {idx} out of range [0, {max_len - 1}]")
        values.append(idx)
    if not values:
        raise ValueError("No valid sample ids provided")
    return values


def upsample_nearest_2x(arr: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def grad_mean(arr: np.ndarray) -> float:
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    g = np.sqrt(gx[:h, :w] ** 2 + gy[:h, :w] ** 2)
    return float(g.mean())


def save_pair_figure(noisy_up: np.ndarray, pred: np.ndarray, out_path: Path, title: str, cmap: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(noisy_up, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title("Noisy LR (2x nearest)")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title("NAFNet Restored")
    axes[1].axis("off")

    diff = np.abs(pred - noisy_up)
    axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("|Restored - NoisyUp|")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_grid(rows: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path, cmap: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (name, noisy_up, pred) in enumerate(rows):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        ax_l.imshow(noisy_up, cmap=cmap, vmin=0.0, vmax=1.0)
        ax_l.set_title(f"{name} noisy-up")
        ax_l.axis("off")

        ax_r.imshow(pred, cmap=cmap, vmin=0.0, vmax=1.0)
        ax_r.set_title(f"{name} restored")
        ax_r.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy predictions found in {pred_dir}")

    sample_ids = parse_sample_ids(args.sample_ids, max_len=len(pred_files))
    rows = []
    stat_lines = ["name\tstd_noisy_up\tstd_restored\tgrad_noisy_up\tgrad_restored"]

    for idx in sample_ids:
        pred_path = pred_files[idx]
        noisy_path = input_dir / pred_path.name
        if not noisy_path.exists():
            raise FileNotFoundError(f"Missing matching input file: {noisy_path}")

        noisy = np.load(noisy_path).astype(np.float32)
        pred = np.load(pred_path).astype(np.float32)
        noisy_up = upsample_nearest_2x(noisy)

        if noisy_up.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {pred_path.name}: noisy_up={noisy_up.shape}, pred={pred.shape}")

        title = f"{pred_path.name}"
        save_pair_figure(noisy_up, pred, out_dir / f"{pred_path.stem}.png", title=title, cmap=args.cmap)

        rows.append((pred_path.name, noisy_up, pred))

        stat_lines.append(
            "\t".join(
                [
                    pred_path.name,
                    f"{float(noisy_up.std()):.6f}",
                    f"{float(pred.std()):.6f}",
                    f"{grad_mean(noisy_up):.6f}",
                    f"{grad_mean(pred):.6f}",
                ]
            )
        )

    save_grid(rows, out_dir / "test_preview_grid.png", cmap=args.cmap)
    (out_dir / "summary.txt").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")

    print(f"Saved {len(rows)} qualitative PNGs + grid to {out_dir}")
    print(f"Saved stats summary to {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()