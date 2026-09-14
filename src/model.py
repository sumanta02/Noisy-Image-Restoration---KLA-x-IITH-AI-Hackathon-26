from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NAFNetH2Config:
    in_channels: int = 1
    out_channels: int = 1
    scale: int = 2
    width: int = 64
    enc_blk_nums: Sequence[int] = (2, 2, 4, 8)
    middle_blk_num: int = 12
    dec_blk_nums: Sequence[int] = (2, 2, 2, 2)
    official_repo: str = "nafnet/official"
    upsample_mode: str = "bilinear"
    clamp_output: bool = True


def _as_int_tuple(v: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in v)


def _resolve_state_dict(payload: object) -> tuple[str, dict]:
    if isinstance(payload, dict):
        if "params_ema" in payload and isinstance(payload["params_ema"], dict):
            return "params_ema", payload["params_ema"]
        if "params" in payload and isinstance(payload["params"], dict):
            return "params", payload["params"]
        if "model" in payload and isinstance(payload["model"], dict):
            return "model", payload["model"]
        if payload and all(isinstance(k, str) for k in payload.keys()):
            if any(torch.is_tensor(v) for v in payload.values()):
                return "root", payload
    raise ValueError("Could not locate a model state dict in checkpoint payload")


def _ensure_official_repo_on_path(official_repo: str) -> Path:
    repo = Path(official_repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(
            f"Official NAFNet repo not found at {repo}. "
            "Clone it first: git clone https://github.com/megvii-research/NAFNet.git nafnet/official"
        )
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _import_official_nafnet(official_repo: str):
    _ensure_official_repo_on_path(official_repo)
    module = importlib.import_module("basicsr.models.archs.NAFNet_arch")
    if not hasattr(module, "NAFNet"):
        raise RuntimeError("Failed to import NAFNet from official repository")
    return getattr(module, "NAFNet")


class NAFNetH2SR(nn.Module):
    """
    Wrapper around official NAFNet for LR noisy -> HR restored.

    Pipeline:
      1) Upsample LR input to HR with interpolation.
      2) Replicate grayscale to RGB for compatibility with official pretrained RGB NAFNet.
      3) Run official NAFNet.
      4) Collapse RGB prediction to grayscale output.
    """

    def __init__(self, cfg: NAFNetH2Config) -> None:
        super().__init__()
        if cfg.scale <= 0:
            raise ValueError("scale must be > 0")
        if cfg.upsample_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("upsample_mode must be one of: nearest/bilinear/bicubic")

        self.cfg = cfg
        self.scale = int(cfg.scale)
        self.in_channels = int(cfg.in_channels)
        self.out_channels = int(cfg.out_channels)
        self.upsample_mode = cfg.upsample_mode
        self.clamp_output = bool(cfg.clamp_output)

        nafnet_cls = _import_official_nafnet(cfg.official_repo)
        self.backbone = nafnet_cls(
            img_channel=3,
            width=int(cfg.width),
            middle_blk_num=int(cfg.middle_blk_num),
            enc_blk_nums=_as_int_tuple(cfg.enc_blk_nums),
            dec_blk_nums=_as_int_tuple(cfg.dec_blk_nums),
        )

    def _upsample(self, y: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            return y
        if self.upsample_mode == "nearest":
            return F.interpolate(y, scale_factor=self.scale, mode=self.upsample_mode)
        return F.interpolate(y, scale_factor=self.scale, mode=self.upsample_mode, align_corners=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape: {tuple(y.shape)}")

        y_up = self._upsample(y)

        if y_up.shape[1] == 1:
            x = y_up.repeat(1, 3, 1, 1)
        elif y_up.shape[1] == 3:
            x = y_up
        else:
            raise ValueError("NAFNetH2SR expects input channels to be 1 or 3")

        out_rgb = self.backbone(x)

        if self.out_channels == 1:
            out = out_rgb.mean(dim=1, keepdim=True)
        elif self.out_channels == 3:
            out = out_rgb
        else:
            raise ValueError("NAFNetH2SR currently supports out_channels=1 or 3")

        if self.clamp_output:
            out = out.clamp(0.0, 1.0)

        h_lr, w_lr = y.shape[-2:]
        return out[:, :, : h_lr * self.scale, : w_lr * self.scale]


def make_nafnet_config(
    preset: str,
    in_channels: int = 1,
    out_channels: int = 1,
    scale: int = 2,
    official_repo: str = "nafnet/official",
    upsample_mode: str = "bilinear",
) -> NAFNetH2Config:
    p = preset.lower()
    if p in {"width32", "sidd-width32", "nafnet-sidd-width32"}:
        return NAFNetH2Config(
            in_channels=in_channels,
            out_channels=out_channels,
            scale=scale,
            width=32,
            enc_blk_nums=(2, 2, 4, 8),
            middle_blk_num=12,
            dec_blk_nums=(2, 2, 2, 2),
            official_repo=official_repo,
            upsample_mode=upsample_mode,
        )
    if p in {"width64", "sidd-width64", "nafnet-sidd-width64"}:
        return NAFNetH2Config(
            in_channels=in_channels,
            out_channels=out_channels,
            scale=scale,
            width=64,
            enc_blk_nums=(2, 2, 4, 8),
            middle_blk_num=12,
            dec_blk_nums=(2, 2, 2, 2),
            official_repo=official_repo,
            upsample_mode=upsample_mode,
        )
    raise ValueError(f"Unknown NAFNet preset: {preset}")


def load_nafnet_pretrained(model: NAFNetH2SR, checkpoint_path: str, strict: bool = False) -> dict:
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {p}")

    payload = torch.load(p, map_location="cpu")
    source, raw_state = _resolve_state_dict(payload)

    state = {}
    for k, v in raw_state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        state[nk] = v

    missing, unexpected = model.backbone.load_state_dict(state, strict=strict)
    matched = len(state) - len(unexpected)

    return {
        "checkpoint": str(p),
        "source": source,
        "strict": bool(strict),
        "matched": int(max(matched, 0)),
        "missing": int(len(missing)),
        "unexpected": int(len(unexpected)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
