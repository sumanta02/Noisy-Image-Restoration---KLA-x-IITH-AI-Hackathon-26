from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


Pair = Tuple[str, str, str]


def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            return arr
        if arr.shape[-1] in (1, 3):
            return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported ndarray shape: {arr.shape}")


def load_npy_image(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    return _to_chw(arr)


def list_train_pairs(gt_dir: str, noisy_dir: str) -> List[Pair]:
    gt_names = {f for f in os.listdir(gt_dir) if f.endswith(".npy")}
    noisy_names = {f for f in os.listdir(noisy_dir) if f.endswith(".npy")}
    common = sorted(gt_names & noisy_names)

    pairs: List[Pair] = []
    for name in common:
        pairs.append((name, os.path.join(gt_dir, name), os.path.join(noisy_dir, name)))
    return pairs


def split_pairs(pairs: Sequence[Pair], val_ratio: float = 0.02, seed: int = 42) -> Tuple[List[Pair], List[Pair]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    idxs = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    n_val = int(round(len(pairs) * val_ratio))
    val_idxs = set(idxs[:n_val])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idxs]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idxs]
    return train_pairs, val_pairs


def _spatially_align(gt: np.ndarray, noisy: np.ndarray, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    h_gt, w_gt = gt.shape[-2:]
    h_lr, w_lr = noisy.shape[-2:]

    h_lr_aligned = min(h_lr, h_gt // scale)
    w_lr_aligned = min(w_lr, w_gt // scale)

    gt = gt[..., : h_lr_aligned * scale, : w_lr_aligned * scale]
    noisy = noisy[..., :h_lr_aligned, :w_lr_aligned]
    return gt, noisy


def _random_crop_pair(gt: np.ndarray, noisy: np.ndarray, patch_size: int, scale: int, enforce_lr_multiple: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    h_gt, w_gt = gt.shape[-2:]

    hr_patch = min(patch_size, h_gt, w_gt)
    base_multiple = max(scale * enforce_lr_multiple, scale)
    hr_patch = (hr_patch // base_multiple) * base_multiple
    if hr_patch == 0:
        hr_patch = (min(h_gt, w_gt) // scale) * scale
    if hr_patch <= 0:
        raise ValueError("Patch size became zero after alignment. Use larger input images.")

    lr_patch = hr_patch // scale
    h_lr, w_lr = noisy.shape[-2:]

    if h_lr == lr_patch:
        top_lr = 0
    else:
        top_lr = random.randint(0, h_lr - lr_patch)

    if w_lr == lr_patch:
        left_lr = 0
    else:
        left_lr = random.randint(0, w_lr - lr_patch)

    top_gt = top_lr * scale
    left_gt = left_lr * scale

    gt = gt[..., top_gt : top_gt + hr_patch, left_gt : left_gt + hr_patch]
    noisy = noisy[..., top_lr : top_lr + lr_patch, left_lr : left_lr + lr_patch]
    return gt, noisy


def _blur3x3_reflect(x: np.ndarray) -> np.ndarray:
    pad_w = np.pad(x, ((0, 0), (0, 0), (1, 1)), mode="reflect")
    tmp = (pad_w[:, :, :-2] + 2.0 * pad_w[:, :, 1:-1] + pad_w[:, :, 2:]) * 0.25

    pad_h = np.pad(tmp, ((0, 0), (1, 1), (0, 0)), mode="reflect")
    out = (pad_h[:, :-2, :] + 2.0 * pad_h[:, 1:-1, :] + pad_h[:, 2:, :]) * 0.25
    return out.astype(np.float32, copy=False)


def _clip_noisy(x: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    return np.clip(x, clip_min, clip_max).astype(np.float32, copy=False)


def _synthetic_ood_degrade(
    noisy: np.ndarray,
    profile: str,
    max_transforms: int,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    if profile == "basic":
        return noisy

    c, h, w = noisy.shape
    if h <= 2 or w <= 2:
        return noisy

    def _op_affine(v: np.ndarray) -> np.ndarray:
        gain = random.uniform(0.75, 1.25)
        bias = random.uniform(-0.08, 0.08)
        return v * gain + bias

    def _op_gamma(v: np.ndarray) -> np.ndarray:
        base = np.clip(v, 0.0, 1.0)
        gamma = random.uniform(0.7, 1.5)
        mapped = np.power(base + 1e-6, gamma)
        blend = random.uniform(0.4, 0.9)
        return blend * mapped + (1.0 - blend) * base

    def _op_gaussian_noise(v: np.ndarray) -> np.ndarray:
        sigma = random.uniform(0.005, 0.06 if profile == "max" else 0.04)
        return v + np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)

    def _op_speckle(v: np.ndarray) -> np.ndarray:
        sigma = random.uniform(0.01, 0.08 if profile == "max" else 0.05)
        return v * (1.0 + np.random.normal(0.0, sigma, size=v.shape).astype(np.float32))

    def _op_poisson(v: np.ndarray) -> np.ndarray:
        peak = random.uniform(20.0, 120.0)
        base = np.clip(v, 0.0, 1.0)
        sampled = np.random.poisson(base * peak).astype(np.float32) / peak
        blend = random.uniform(0.5, 0.9)
        return blend * sampled + (1.0 - blend) * base

    def _op_blur(v: np.ndarray) -> np.ndarray:
        out = _blur3x3_reflect(v)
        if random.random() < 0.4:
            out = _blur3x3_reflect(out)
        return out

    def _op_stripe(v: np.ndarray) -> np.ndarray:
        amp = random.uniform(0.005, 0.04 if profile == "max" else 0.025)
        if random.random() < 0.5:
            stripe = np.random.normal(0.0, amp, size=(1, h, 1)).astype(np.float32)
        else:
            stripe = np.random.normal(0.0, amp, size=(1, 1, w)).astype(np.float32)
        return v + stripe

    def _op_impulse(v: np.ndarray) -> np.ndarray:
        ratio = random.uniform(0.001, 0.02 if profile == "max" else 0.008)
        mask = np.random.rand(1, h, w) < ratio
        vals = np.random.uniform(clip_min, clip_max, size=(1, h, w)).astype(np.float32)
        return np.where(mask, vals, v)

    def _op_cutout(v: np.ndarray) -> np.ndarray:
        out = v.copy()
        holes = random.randint(1, 3 if profile == "max" else 2)
        for _ in range(holes):
            hh = random.randint(max(4, h // 20), max(8, h // 5))
            ww = random.randint(max(4, w // 20), max(8, w // 5))
            top = random.randint(0, max(h - hh, 0))
            left = random.randint(0, max(w - ww, 0))
            fill = float(np.random.uniform(clip_min, clip_max))
            out[:, top : top + hh, left : left + ww] = fill
        return out

    ops = [_op_affine, _op_gamma, _op_gaussian_noise, _op_speckle, _op_poisson, _op_blur, _op_stripe, _op_impulse, _op_cutout]

    if profile == "max":
        n_ops = random.randint(2, max(2, max_transforms))
    else:
        n_ops = random.randint(1, max(1, min(max_transforms, 3)))

    order = random.sample(ops, k=min(n_ops, len(ops)))
    out = noisy.astype(np.float32, copy=False)
    for op in order:
        out = _clip_noisy(op(out), clip_min=clip_min, clip_max=clip_max)
    return out


def _augment_pair(
    gt: np.ndarray,
    noisy: np.ndarray,
    augment_profile: str,
    ood_prob: float,
    ood_max_transforms: int,
    ood_clip_min: float,
    ood_clip_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        gt = np.flip(gt, axis=-1)
        noisy = np.flip(noisy, axis=-1)

    if random.random() < 0.5:
        gt = np.flip(gt, axis=-2)
        noisy = np.flip(noisy, axis=-2)

    k = random.randint(0, 3)
    if k:
        gt = np.rot90(gt, k=k, axes=(-2, -1))
        noisy = np.rot90(noisy, k=k, axes=(-2, -1))

    if augment_profile != "basic" and random.random() < ood_prob:
        noisy = _synthetic_ood_degrade(
            noisy,
            profile=augment_profile,
            max_transforms=ood_max_transforms,
            clip_min=ood_clip_min,
            clip_max=ood_clip_max,
        )

    return gt.copy(), noisy.copy()


@dataclass(frozen=True)
class TrainDatasetConfig:
    scale: int = 2
    patch_size: int = 256
    augment: bool = True
    augment_profile: str = "basic"  # basic | strong | max
    ood_prob: float = 0.0
    ood_max_transforms: int = 4
    ood_clip_min: float = -0.25
    ood_clip_max: float = 1.80


class NpyPairDataset(Dataset):
    """Paired GT/NoisyLR dataset loaded from .npy files."""

    def __init__(self, pairs: Sequence[Pair], cfg: TrainDatasetConfig, training: bool) -> None:
        self.pairs = list(pairs)
        self.cfg = cfg
        self.training = training
        self.augment_profile = cfg.augment_profile
        self.ood_prob = float(cfg.ood_prob)
        self.ood_max_transforms = int(cfg.ood_max_transforms)
        self.ood_clip_min = float(cfg.ood_clip_min)
        self.ood_clip_max = float(cfg.ood_clip_max)

    def set_ood_policy(self, profile: str, prob: float, max_transforms: int) -> None:
        self.augment_profile = str(profile)
        self.ood_prob = float(max(0.0, min(1.0, prob)))
        self.ood_max_transforms = int(max(1, max_transforms))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        name, gt_path, noisy_path = self.pairs[index]
        gt = load_npy_image(gt_path)
        noisy = load_npy_image(noisy_path)

        gt, noisy = _spatially_align(gt, noisy, scale=self.cfg.scale)

        if self.training and self.cfg.patch_size > 0:
            gt, noisy = _random_crop_pair(gt, noisy, patch_size=self.cfg.patch_size, scale=self.cfg.scale)

        if self.training and self.cfg.augment:
            gt, noisy = _augment_pair(
                gt,
                noisy,
                augment_profile=self.augment_profile,
                ood_prob=self.ood_prob,
                ood_max_transforms=self.ood_max_transforms,
                ood_clip_min=self.ood_clip_min,
                ood_clip_max=self.ood_clip_max,
            )

        # GT is normalized in [0,1] by dataset definition, but clipping adds numerical safety.
        gt = np.clip(gt, 0.0, 1.0)

        return {
            "name": name,
            "gt": torch.from_numpy(np.ascontiguousarray(gt)),
            "noisy": torch.from_numpy(np.ascontiguousarray(noisy)),
        }


class NpyNoisyDataset(Dataset):
    """Noisy-only dataset for inference; expects .npy tensors in LR space."""

    def __init__(self, noisy_dir: str) -> None:
        self.noisy_dir = noisy_dir
        self.names = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".npy")])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        path = os.path.join(self.noisy_dir, name)
        noisy = load_npy_image(path)
        return {
            "name": name,
            "noisy": torch.from_numpy(np.ascontiguousarray(noisy)),
        }
