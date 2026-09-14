from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            return arr
        if arr.shape[-1] in (1, 3):
            return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported ndarray shape: {arr.shape}")


def _spatially_align(gt: np.ndarray, noisy: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    h_gt, w_gt = gt.shape[-2:]
    h_lr, w_lr = noisy.shape[-2:]

    h_lr_aligned = min(h_lr, h_gt // scale)
    w_lr_aligned = min(w_lr, w_gt // scale)

    gt = gt[..., : h_lr_aligned * scale, : w_lr_aligned * scale]
    noisy = noisy[..., :h_lr_aligned, :w_lr_aligned]
    return gt, noisy


def _to_uint8_image(arr_chw: np.ndarray, out_channels: int) -> Image.Image:
    arr_chw = np.clip(arr_chw.astype(np.float32), 0.0, 1.0)

    if out_channels == 1:
        if arr_chw.shape[0] == 1:
            hw = arr_chw[0]
        elif arr_chw.shape[0] == 3:
            hw = np.mean(arr_chw, axis=0)
        else:
            raise ValueError(f"Unsupported channels for grayscale export: {arr_chw.shape[0]}")
        data = np.round(hw * 255.0).astype(np.uint8)
        return Image.fromarray(data, mode="L")

    if out_channels == 3:
        if arr_chw.shape[0] == 1:
            arr_chw = np.repeat(arr_chw, 3, axis=0)
        elif arr_chw.shape[0] != 3:
            raise ValueError(f"Unsupported channels for RGB export: {arr_chw.shape[0]}")
        data = np.round(np.transpose(arr_chw, (1, 2, 0)) * 255.0).astype(np.uint8)
        return Image.fromarray(data, mode="RGB")

    raise ValueError("out_channels must be 1 or 3")


def _upsample_bicubic_lr_to_hr(noisy_chw: np.ndarray, scale: int) -> np.ndarray:
    c, h, w = noisy_chw.shape
    out = []
    for ci in range(c):
        pil = Image.fromarray(np.round(np.clip(noisy_chw[ci], 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        pil = pil.resize((w * scale, h * scale), resample=Image.BICUBIC)
        out.append(np.asarray(pil, dtype=np.float32) / 255.0)
    return np.stack(out, axis=0)


def _save_split(
    names: list[str],
    gt_dir: Path,
    noisy_dir: Path,
    out_root: Path,
    split: str,
    scale: int,
    out_channels: int,
) -> None:
    inp_lr_dir = out_root / split / "input_lr"
    inp_up_dir = out_root / split / "input_up"
    target_dir = out_root / split / "target"

    inp_lr_dir.mkdir(parents=True, exist_ok=True)
    inp_up_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        gt = _to_chw(np.load(gt_dir / name).astype(np.float32))
        noisy = _to_chw(np.load(noisy_dir / name).astype(np.float32))

        gt, noisy = _spatially_align(gt, noisy, scale=scale)
        noisy_up = _upsample_bicubic_lr_to_hr(noisy, scale=scale)

        gt_img = _to_uint8_image(gt, out_channels=out_channels)
        lr_img = _to_uint8_image(noisy, out_channels=out_channels)
        up_img = _to_uint8_image(noisy_up, out_channels=out_channels)

        stem = Path(name).stem
        gt_img.save(target_dir / f"{stem}.png")
        lr_img.save(inp_lr_dir / f"{stem}.png")
        up_img.save(inp_up_dir / f"{stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BasicSR-friendly PNG pairs from NPY train data")
    parser.add_argument("--train-root", type=str, default="data/train")
    parser.add_argument("--gt-subdir", type=str, default="GT")
    parser.add_argument("--noisy-subdir", type=str, default="NoisyLR")
    parser.add_argument("--out-root", type=str, default="external/restormer_data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--out-channels", type=int, default=3, choices=[1, 3])
    args = parser.parse_args()

    gt_dir = Path(args.train_root) / args.gt_subdir
    noisy_dir = Path(args.train_root) / args.noisy_subdir
    out_root = Path(args.out_root)

    gt_names = {p.name for p in gt_dir.glob("*.npy")}
    noisy_names = {p.name for p in noisy_dir.glob("*.npy")}
    names = sorted(gt_names & noisy_names)

    if not names:
        raise RuntimeError(f"No paired .npy files found in {gt_dir} and {noisy_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(names)

    n_val = int(round(len(names) * args.val_ratio))
    val_names = names[:n_val]
    train_names = names[n_val:]

    _save_split(
        names=train_names,
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        out_root=out_root,
        split="train",
        scale=args.scale,
        out_channels=args.out_channels,
    )
    _save_split(
        names=val_names,
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        out_root=out_root,
        split="val",
        scale=args.scale,
        out_channels=args.out_channels,
    )

    print(f"Prepared dataset at: {out_root}")
    print(f"Train pairs: {len(train_names)}")
    print(f"Val pairs:   {len(val_names)}")
    print("Subfolders: train/input_lr, train/input_up, train/target, val/input_lr, val/input_up, val/target")


if __name__ == "__main__":
    main()
