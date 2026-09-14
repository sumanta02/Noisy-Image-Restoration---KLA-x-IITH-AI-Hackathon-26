from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create qualitative comparison panels between two prediction folders")
    parser.add_argument("--input-dir", type=str, default="data/test/NoisyLR")
    parser.add_argument("--pred-dir-a", type=str, required=True)
    parser.add_argument("--pred-dir-b", type=str, required=True)
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--output-dir", type=str, required=True)
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


def save_grid(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], out_path: Path, cmap: str, label_a: str, label_b: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (name, noisy_up, pred_a, pred_b) in enumerate(rows):
        ax_l = axes[i, 0]
        ax_m = axes[i, 1]
        ax_r = axes[i, 2]

        ax_l.imshow(noisy_up, cmap=cmap, vmin=0.0, vmax=1.0)
        ax_l.set_title(f"{name} noisy-up")
        ax_l.axis("off")

        ax_m.imshow(pred_a, cmap=cmap, vmin=0.0, vmax=1.0)
        ax_m.set_title(f"{name} {label_a}")
        ax_m.axis("off")

        ax_r.imshow(pred_b, cmap=cmap, vmin=0.0, vmax=1.0)
        ax_r.set_title(f"{name} {label_b}")
        ax_r.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_pair_figure(
    name: str,
    noisy_up: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    out_path: Path,
    cmap: str,
    label_a: str,
    label_b: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(noisy_up, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title("Noisy LR (2x nearest)")
    axes[0].axis("off")

    axes[1].imshow(pred_a, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(label_a)
    axes[1].axis("off")

    axes[2].imshow(pred_b, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[2].set_title(label_b)
    axes[2].axis("off")

    diff = np.abs(pred_a - pred_b)
    axes[3].imshow(diff, cmap="magma")
    axes[3].set_title(f"|{label_a} - {label_b}|")
    axes[3].axis("off")

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir)
    pred_dir_a = Path(args.pred_dir_a)
    pred_dir_b = Path(args.pred_dir_b)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files_a = sorted(pred_dir_a.glob("*.npy"))
    pred_files_b = sorted(pred_dir_b.glob("*.npy"))
    if not pred_files_a:
        raise FileNotFoundError(f"No .npy predictions found in {pred_dir_a}")
    if not pred_files_b:
        raise FileNotFoundError(f"No .npy predictions found in {pred_dir_b}")

    names_b = {p.name for p in pred_files_b}
    missing = [p.name for p in pred_files_a if p.name not in names_b]
    if missing:
        raise ValueError(f"Missing {len(missing)} names in {pred_dir_b}; first missing: {missing[0]}")

    sample_ids = parse_sample_ids(args.sample_ids, max_len=len(pred_files_a))
    rows = []
    stat_lines = [
        "name"
        "\tstd_noisy_up"
        "\tstd_a"
        "\tstd_b"
        "\tstd_b_minus_a"
        "\tgrad_noisy_up"
        "\tgrad_a"
        "\tgrad_b"
        "\tgrad_b_minus_a"
    ]

    std_delta = []
    grad_delta = []

    for idx in sample_ids:
        pred_a_path = pred_files_a[idx]
        pred_b_path = pred_dir_b / pred_a_path.name
        noisy_path = input_dir / pred_a_path.name

        if not noisy_path.exists():
            raise FileNotFoundError(f"Missing matching input file: {noisy_path}")

        noisy = np.load(noisy_path).astype(np.float32)
        pred_a = np.load(pred_a_path).astype(np.float32)
        pred_b = np.load(pred_b_path).astype(np.float32)
        noisy_up = upsample_nearest_2x(noisy)

        if noisy_up.shape != pred_a.shape or pred_a.shape != pred_b.shape:
            raise ValueError(
                f"Shape mismatch for {pred_a_path.name}: noisy_up={noisy_up.shape}, a={pred_a.shape}, b={pred_b.shape}"
            )

        save_pair_figure(
            name=pred_a_path.name,
            noisy_up=noisy_up,
            pred_a=pred_a,
            pred_b=pred_b,
            out_path=out_dir / f"{pred_a_path.stem}.png",
            cmap=args.cmap,
            label_a=args.label_a,
            label_b=args.label_b,
        )

        rows.append((pred_a_path.name, noisy_up, pred_a, pred_b))

        s_noisy = float(noisy_up.std())
        s_a = float(pred_a.std())
        s_b = float(pred_b.std())
        g_noisy = grad_mean(noisy_up)
        g_a = grad_mean(pred_a)
        g_b = grad_mean(pred_b)

        std_delta.append(s_b - s_a)
        grad_delta.append(g_b - g_a)

        stat_lines.append(
            "\t".join(
                [
                    pred_a_path.name,
                    f"{s_noisy:.6f}",
                    f"{s_a:.6f}",
                    f"{s_b:.6f}",
                    f"{(s_b - s_a):.6f}",
                    f"{g_noisy:.6f}",
                    f"{g_a:.6f}",
                    f"{g_b:.6f}",
                    f"{(g_b - g_a):.6f}",
                ]
            )
        )

    save_grid(rows, out_dir / "comparison_grid.png", cmap=args.cmap, label_a=args.label_a, label_b=args.label_b)

    stat_lines.extend(
        [
            "",
            "aggregate\tvalue",
            f"samples\t{len(sample_ids)}",
            f"mean_std_b_minus_a\t{float(np.mean(std_delta)):.6f}",
            f"mean_grad_b_minus_a\t{float(np.mean(grad_delta)):.6f}",
            f"positive_grad_delta_fraction\t{float(np.mean(np.array(grad_delta) > 0.0)):.6f}",
        ]
    )
    (out_dir / "summary.txt").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")

    print(f"Saved {len(rows)} per-sample comparison PNGs + grid to {out_dir}")
    print(f"Saved comparison stats summary to {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()