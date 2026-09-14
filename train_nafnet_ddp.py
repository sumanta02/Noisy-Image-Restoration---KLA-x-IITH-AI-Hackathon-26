from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.model import NAFNetH2SR, load_nafnet_pretrained, make_nafnet_config
from src.config import apply_config_defaults, load_config_file
from src.dataset import NpyPairDataset, TrainDatasetConfig, list_train_pairs, split_pairs
from src.ddp import (
    barrier,
    cleanup_distributed,
    cuda_environment_summary,
    init_distributed_mode,
    is_main_process,
    reduce_dict,
)
from src.losses import (
    HAS_LPIPS,
    HAS_MSSSIM,
    LossConfig,
    build_lpips_model,
    combined_restoration_loss,
    forward_consistency_h2,
    heteroscedastic_variance_h2,
    lpips_loss,
    ssim_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train official NAFNet (author implementation) on H2 data under DDP")

    parser.add_argument("--config", type=str, default="", help="Path to YAML/JSON config file")

    parser.add_argument("--data-root", type=str, default="../data/train")
    parser.add_argument("--gt-subdir", type=str, default="GT")
    parser.add_argument("--noisy-subdir", type=str, default="NoisyLR")
    parser.add_argument("--output-dir", type=str, default="outputs/nafnet_h2")

    parser.add_argument("--nafnet-preset", type=str, default="sidd-width64", choices=["sidd-width32", "sidd-width64"])
    parser.add_argument("--official-repo", type=str, default="nafnet/official")
    parser.add_argument("--pretrained", type=str, default="nafnet/weights/nafnet_sidd_width64.pth")
    parser.add_argument("--strict-pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--staged-freeze",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply progressive freezing of early encoder stages during finetuning",
    )
    parser.add_argument(
        "--freeze-stage1-end-epoch",
        type=int,
        default=10,
        help="Epoch where phase-1 freezing ends (phase-1 active for epoch < value)",
    )
    parser.add_argument(
        "--freeze-stage2-end-epoch",
        type=int,
        default=25,
        help="Epoch where phase-2 partial freeze ends (phase-2 active for epoch < value)",
    )
    parser.add_argument(
        "--freeze-intro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep NAFNet intro stem frozen during staged-freeze phases",
    )
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--upsample-mode", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic"])

    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--optim-beta1", type=float, default=0.9)
    parser.add_argument("--optim-beta2", type=float, default=0.9)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-warmup-start-factor", type=float, default=0.05)
    parser.add_argument("--lr-eta-min", type=float, default=1e-7)
    parser.add_argument(
        "--total-iters",
        type=int,
        default=400000,
        help="Total optimizer updates for cosine schedule; set <=0 to use epochs*steps_per_epoch.",
    )

    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--augment-profile", type=str, default="max", choices=["basic", "strong", "max"])
    parser.add_argument("--ood-prob", type=float, default=0.95)
    parser.add_argument("--ood-max-transforms", type=int, default=5)
    parser.add_argument("--ood-clip-min", type=float, default=-0.25)
    parser.add_argument("--ood-clip-max", type=float, default=1.8)
    parser.add_argument("--augment-curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--finetune-preset", type=str, default="roi_psnr_120e", choices=["none", "roi_psnr_120e"])
    parser.add_argument("--stage1-end-epoch", type=int, default=20)
    parser.add_argument("--stage2-end-epoch", type=int, default=90)

    parser.add_argument("--lambda-psnr", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--lambda-charbonnier", type=float, default=0.0)
    parser.add_argument("--lambda-fft", type=float, default=0.0)

    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--dc-schedule", type=str, default="constant", choices=["adaptive", "linear", "constant"])
    parser.add_argument("--dc-lambda-start", type=float, default=0.005)
    parser.add_argument("--dc-lambda-step", type=float, default=0.005)
    parser.add_argument("--dc-lambda-cap", type=float, default=0.04)
    parser.add_argument("--dc-patience", type=int, default=2)
    parser.add_argument("--dc-min-delta", type=float, default=1e-4)
    parser.add_argument("--dc-ramp-epochs", type=int, default=10)

    parser.add_argument("--lambda-ssim", type=float, default=0.05)
    parser.add_argument("--ssim-start-epoch", type=int, default=60)
    parser.add_argument("--ssim-lambda-start", type=float, default=0.0005)
    parser.add_argument("--ssim-ramp-epochs", type=int, default=100)
    parser.add_argument("--ssim-contribute-to-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pixel-loss-type", type=str, default="psnr", choices=["l1", "charbonnier", "psnr"])
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--lambda-edge", type=float, default=0.0)

    parser.add_argument("--lambda-dc", type=float, default=0.03)
    parser.add_argument("--student-t-nu", type=float, default=3.0)

    parser.add_argument("--lambda-lpips-max", type=float, default=0.0)
    parser.add_argument("--lpips-start-epoch", type=int, default=60)
    parser.add_argument("--lpips-lambda-start", type=float, default=0.001)
    parser.add_argument("--lpips-ramp-epochs", type=int, default=30)
    parser.add_argument("--lpips-gate-psnr", type=float, default=25.5)
    parser.add_argument("--lpips-gate-min-epoch", type=int, default=40)
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg"])

    parser.add_argument("--require-ssim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dc-debug-first-epoch", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--h2-a", type=float, default=0.14160734)
    parser.add_argument("--h2-b", type=float, default=6.3383e-05)
    parser.add_argument("--h2-jitter", type=float, default=0.025)
    parser.add_argument("--h2-jitter-decay", type=str, default="none", choices=["none", "linear", "cosine"])
    parser.add_argument("--h2-jitter-min", type=float, default=0.025)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save periodic checkpoint every N epochs; set <=0 to disable periodic saves.",
    )
    parser.add_argument("--export-best-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-latest-infer", action=argparse.BooleanOptionalAction, default=True)

    return parser


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    known, _ = pre.parse_known_args()

    parser = build_parser()
    if known.config:
        cfg_data = load_config_file(known.config)
        unknown = apply_config_defaults(parser, cfg_data)
        if unknown:
            print(f"[Config] Ignored unknown keys: {sorted(list(unknown.keys()))}")

    return parser.parse_args()


def resolve_resume_path(args: argparse.Namespace, out_dir: Path) -> Path | None:
    if args.resume:
        p = Path(args.resume)
        if not p.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {p}")
        return p

    if args.auto_resume:
        latest = out_dir / "latest.pt"
        if latest.exists():
            return latest

    return None


class DCScheduler:
    def __init__(
        self,
        start: float = 0.005,
        step: float = 0.005,
        cap: float = 0.05,
        patience: int = 2,
        min_delta: float = 1e-4,
    ) -> None:
        self.lam = float(max(0.0, min(start, cap)))
        self.step = float(step)
        self.cap = float(cap)
        self.patience = int(patience)
        self.min_delta = float(max(0.0, min_delta))

        self._no_improve = 0
        self._best_psnr = -1e9

    def update(self, val_psnr: float) -> float:
        if val_psnr > self._best_psnr + self.min_delta:
            self._best_psnr = float(val_psnr)
            self._no_improve = 0
            self.lam = min(self.cap, self.lam + self.step)
        else:
            self._no_improve += 1
            if self._no_improve >= self.patience:
                self.lam = max(self.step, self.lam - self.step)
                self._no_improve = 0
        return self.lam

    def state_dict(self) -> dict:
        return {
            "lam": self.lam,
            "step": self.step,
            "cap": self.cap,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "_no_improve": self._no_improve,
            "_best_psnr": self._best_psnr,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.lam = float(state.get("lam", self.lam))
        self.step = float(state.get("step", self.step))
        self.cap = float(state.get("cap", self.cap))
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self._no_improve = int(state.get("_no_improve", self._no_improve))
        self._best_psnr = float(state.get("_best_psnr", self._best_psnr))


def build_infer_export_payload(save_payload: dict) -> dict:
    return {
        "epoch": save_payload["epoch"],
        "model": save_payload["model"],
        "best_psnr": save_payload.get("best_psnr", -1e9),
        "best_lpips": save_payload.get("best_lpips", float("inf")),
        "args": save_payload.get("args", {}),
    }


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def effective_h2_jitter(args: argparse.Namespace, epoch: int) -> float:
    base = float(max(args.h2_jitter, 0.0))
    if base <= 0.0:
        return 0.0
    if args.h2_jitter_decay == "none":
        return base

    if args.epochs <= 1:
        return max(float(args.h2_jitter_min), base)

    progress = min(max(epoch / float(args.epochs - 1), 0.0), 1.0)
    if args.h2_jitter_decay == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        factor = 1.0 - progress

    return max(float(args.h2_jitter_min), base * factor)


def effective_lambda_ssim(args: argparse.Namespace, epoch: int) -> float:
    if epoch < args.ssim_start_epoch:
        return 0.0

    lam_cap = float(max(args.lambda_ssim, 0.0))
    lam_start = float(max(args.ssim_lambda_start, 0.0))
    lam_start = min(lam_start, lam_cap)

    if args.ssim_ramp_epochs <= 0:
        return lam_cap

    progress = min(max((epoch - args.ssim_start_epoch) / float(args.ssim_ramp_epochs), 0.0), 1.0)
    return lam_start + (lam_cap - lam_start) * progress


def effective_lambda_lpips(args: argparse.Namespace, epoch: int, lpips_gate_epoch: int | None) -> float:
    lam_cap = float(max(args.lambda_lpips_max, 0.0))
    if lam_cap <= 0.0:
        return 0.0
    if lpips_gate_epoch is None or epoch < lpips_gate_epoch:
        return 0.0

    lam_start = float(max(args.lpips_lambda_start, 0.0))
    lam_start = min(lam_start, lam_cap)

    if args.lpips_ramp_epochs <= 0:
        return lam_cap

    progress = min(max((epoch - lpips_gate_epoch) / float(args.lpips_ramp_epochs), 0.0), 1.0)
    return lam_start + (lam_cap - lam_start) * progress


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp_triplet(epoch: int, e1: int, e2: int, e_end: int, a: float, b: float, c: float) -> float:
    if epoch < e1:
        return float(a)
    if epoch < e2:
        t = (epoch - e1) / float(max(e2 - e1, 1))
        return float(_lerp(a, b, t))

    tail = max(e_end - e2 - 1, 1)
    t = min(max((epoch - e2) / float(tail), 0.0), 1.0)
    return float(_lerp(b, c, t))


def effective_stage_weights(args: argparse.Namespace, epoch: int, max_epochs: int) -> dict[str, float]:
    if args.finetune_preset != "roi_psnr_120e":
        return {
            "lambda_psnr": float(args.lambda_psnr),
            "lambda_l1": float(args.lambda_l1),
            "lambda_l2": float(args.lambda_l2),
            "lambda_charbonnier": float(args.lambda_charbonnier),
            "lambda_fft": float(args.lambda_fft),
            "lambda_dc": float(args.lambda_dc),
            "lambda_ssim": float(args.lambda_ssim),
        }

    e1 = max(int(args.stage1_end_epoch), 0)
    e2 = max(int(args.stage2_end_epoch), e1)
    e_end = max(int(max_epochs), e2 + 1)

    return {
        "lambda_psnr": _interp_triplet(epoch, e1, e2, e_end, 1.0, 1.0, 1.0),
        "lambda_l1": _interp_triplet(epoch, e1, e2, e_end, 0.05, 0.03, 0.02),
        "lambda_l2": _interp_triplet(epoch, e1, e2, e_end, 0.01, 0.005, 0.0),
        "lambda_charbonnier": _interp_triplet(epoch, e1, e2, e_end, 0.01, 0.008, 0.005),
        "lambda_fft": _interp_triplet(epoch, e1, e2, e_end, 0.0, 0.01, 0.015),
        "lambda_dc": _interp_triplet(epoch, e1, e2, e_end, 0.0, 0.01, 0.008),
        "lambda_ssim": _interp_triplet(epoch, e1, e2, e_end, 0.0, 0.015, 0.02),
    }


def effective_augment_policy(args: argparse.Namespace, epoch: int, max_epochs: int) -> dict[str, float | int | str]:
    max_prob = float(max(0.0, min(1.0, args.ood_prob)))
    max_ops = int(max(1, args.ood_max_transforms))
    profile = str(args.augment_profile)

    if not args.augment_curriculum:
        return {
            "profile": profile,
            "ood_prob": max_prob,
            "ood_max_transforms": max_ops,
        }

    e1 = max(int(args.stage1_end_epoch), 0)
    e2 = max(int(args.stage2_end_epoch), e1)
    e_end = max(int(max_epochs), e2 + 1)

    if epoch < e1:
        return {
            "profile": "strong" if profile == "max" else profile,
            "ood_prob": min(max_prob, 0.70),
            "ood_max_transforms": min(max_ops, 3),
        }

    if epoch < e2:
        return {
            "profile": profile,
            "ood_prob": max_prob,
            "ood_max_transforms": max_ops,
        }

    tail = max(e_end - e2 - 1, 1)
    t = min(max((epoch - e2) / float(tail), 0.0), 1.0)
    prob_start = min(max_prob, 0.80)
    prob_end = min(max_prob, 0.50)
    ops_start = min(max_ops, 3)
    ops_end = min(max_ops, 2)

    return {
        "profile": "strong" if profile == "max" else profile,
        "ood_prob": _lerp(prob_start, prob_end, t),
        "ood_max_transforms": int(round(_lerp(float(ops_start), float(ops_end), t))),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
    lpips_net: torch.nn.Module | None,
) -> dict:
    model.eval()

    count = 0
    loss_l1 = 0.0
    loss_mse = 0.0
    metric_ssim = 0.0
    metric_lpips = 0.0

    for batch in loader:
        y = batch["noisy"].to(device, non_blocking=True)
        x = batch["gt"].to(device, non_blocking=True)

        pred = model(y)
        l1 = F.l1_loss(pred, x)
        mse = F.mse_loss(pred, x)
        ssim_score = 1.0 - ssim_loss(pred, x) if HAS_MSSSIM else torch.zeros((), device=device, dtype=pred.dtype)
        lpips_score = lpips_loss(pred, x, lpips_net=lpips_net)

        bsz = y.shape[0]
        count += bsz
        loss_l1 += float(l1.item()) * bsz
        loss_mse += float(mse.item()) * bsz
        metric_ssim += float(ssim_score.item()) * bsz
        metric_lpips += float(lpips_score.item()) * bsz

    metrics = {
        "val_l1": loss_l1 / max(count, 1),
        "val_mse": loss_mse / max(count, 1),
        "val_ssim": metric_ssim / max(count, 1),
        "val_lpips": metric_lpips / max(count, 1),
    }
    metrics["val_psnr"] = float(10.0 * np.log10(1.0 / max(metrics["val_mse"], 1e-12)))
    return reduce_dict(metrics, device=device, distributed=distributed)


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
    persistent_workers: bool,
):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return loader, sampler


def _all_ranks_finite(t: torch.Tensor, distributed: bool) -> bool:
    finite_local = torch.tensor(1 if torch.isfinite(t).all() else 0, device=t.device, dtype=torch.int32)
    if distributed:
        dist.all_reduce(finite_local, op=dist.ReduceOp.MIN)
    return bool(finite_local.item() == 1)


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(trainable)


def _freeze_phase_for_epoch(args: argparse.Namespace, epoch: int) -> str:
    if not args.staged_freeze:
        return "all_trainable"
    if epoch < args.freeze_stage1_end_epoch:
        return "freeze_early"
    if epoch < args.freeze_stage2_end_epoch:
        return "freeze_partial"
    return "all_trainable"


def _apply_freeze_phase(model: torch.nn.Module, args: argparse.Namespace, phase: str) -> tuple[int, int]:
    core = model.module if isinstance(model, DDP) else model
    backbone = getattr(core, "backbone", None)
    if backbone is None:
        total = sum(p.numel() for p in core.parameters())
        trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
        return trainable, total

    # Reset to fully trainable first, then freeze selected stages for current phase.
    _set_module_trainable(backbone, True)
    if phase == "freeze_early":
        if args.freeze_intro:
            _set_module_trainable(getattr(backbone, "intro", None), False)
        encoders = getattr(backbone, "encoders", None)
        downs = getattr(backbone, "downs", None)
        if encoders is not None:
            if len(encoders) > 0:
                _set_module_trainable(encoders[0], False)
            if len(encoders) > 1:
                _set_module_trainable(encoders[1], False)
        if downs is not None:
            if len(downs) > 0:
                _set_module_trainable(downs[0], False)
            if len(downs) > 1:
                _set_module_trainable(downs[1], False)
    elif phase == "freeze_partial":
        if args.freeze_intro:
            _set_module_trainable(getattr(backbone, "intro", None), False)
        encoders = getattr(backbone, "encoders", None)
        downs = getattr(backbone, "downs", None)
        if encoders is not None and len(encoders) > 0:
            _set_module_trainable(encoders[0], False)
        if downs is not None and len(downs) > 0:
            _set_module_trainable(downs[0], False)

    total = sum(p.numel() for p in core.parameters())
    trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    return trainable, total


def main() -> None:
    args = parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated where possible.*",
        category=UserWarning,
        module=r"torch\.optim\.lr_scheduler",
    )

    if args.freeze_stage1_end_epoch < 0:
        raise ValueError("freeze_stage1_end_epoch must be >= 0")
    if args.freeze_stage2_end_epoch < 0:
        raise ValueError("freeze_stage2_end_epoch must be >= 0")
    if args.freeze_stage2_end_epoch < args.freeze_stage1_end_epoch:
        raise ValueError("freeze_stage2_end_epoch must be >= freeze_stage1_end_epoch")

    if args.ssim_start_epoch < 0:
        raise ValueError("ssim_start_epoch must be >= 0")
    if args.ssim_ramp_epochs < 0:
        raise ValueError("ssim_ramp_epochs must be >= 0")
    if args.ssim_lambda_start < 0.0:
        raise ValueError("ssim_lambda_start must be >= 0")
    if args.lambda_ssim > 0.0 and args.ssim_lambda_start > args.lambda_ssim:
        raise ValueError("ssim_lambda_start must be <= lambda_ssim")
    if args.charbonnier_eps <= 0:
        raise ValueError("charbonnier_eps must be > 0")
    if args.lambda_edge < 0:
        raise ValueError("lambda_edge must be >= 0")
    if args.lambda_lpips_max < 0.0:
        raise ValueError("lambda_lpips_max must be >= 0")
    if args.lpips_start_epoch < 0:
        raise ValueError("lpips_start_epoch must be >= 0")
    if args.lpips_gate_min_epoch < 0:
        raise ValueError("lpips_gate_min_epoch must be >= 0")
    if args.lpips_ramp_epochs < 0:
        raise ValueError("lpips_ramp_epochs must be >= 0")
    if args.lpips_lambda_start < 0.0:
        raise ValueError("lpips_lambda_start must be >= 0")
    if args.lpips_lambda_start > args.lambda_lpips_max:
        raise ValueError("lpips_lambda_start must be <= lambda_lpips_max")
    if args.h2_jitter_min < 0.0:
        raise ValueError("h2_jitter_min must be >= 0")
    if args.h2_jitter_min > args.h2_jitter:
        raise ValueError("h2_jitter_min must be <= h2_jitter")
    if not (0.0 <= args.optim_beta1 < 1.0):
        raise ValueError("optim_beta1 must be in [0, 1)")
    if not (0.0 <= args.optim_beta2 < 1.0):
        raise ValueError("optim_beta2 must be in [0, 1)")
    if args.lr_eta_min < 0.0:
        raise ValueError("lr_eta_min must be >= 0")
    if not (0.0 <= args.ood_prob <= 1.0):
        raise ValueError("ood_prob must be in [0, 1]")
    if args.ood_max_transforms <= 0:
        raise ValueError("ood_max_transforms must be > 0")
    if args.ood_clip_min >= args.ood_clip_max:
        raise ValueError("ood_clip_min must be < ood_clip_max")
    if args.stage1_end_epoch < 0:
        raise ValueError("stage1_end_epoch must be >= 0")
    if args.stage2_end_epoch < args.stage1_end_epoch:
        raise ValueError("stage2_end_epoch must be >= stage1_end_epoch")
    for name in ("lambda_psnr", "lambda_l1", "lambda_l2", "lambda_charbonnier", "lambda_fft"):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be >= 0")

    env = init_distributed_mode()
    rank = env["rank"]
    world_size = env["world_size"]
    local_rank = env["local_rank"]
    distributed = env["distributed"]
    device = env["device"]

    seed_everything(args.seed, rank)

    out_dir = Path(args.output_dir)
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    barrier(distributed)

    if is_main_process(rank):
        print("CUDA environment:", json.dumps(cuda_environment_summary(), indent=2))
        print(f"Distributed: {distributed}, world_size={world_size}, local_rank={local_rank}")

    if args.lambda_ssim > 0 and not HAS_MSSSIM:
        msg = "lambda_ssim > 0 but pytorch_msssim is not available. Install with: pip install pytorch-msssim"
        if args.require_ssim:
            raise RuntimeError(msg)
        if is_main_process(rank):
            print(f"[Warning] {msg}. SSIM term will be zero.")

    gt_dir = os.path.join(args.data_root, args.gt_subdir)
    noisy_dir = os.path.join(args.data_root, args.noisy_subdir)
    pairs = list_train_pairs(gt_dir, noisy_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No paired .npy files found in {gt_dir} and {noisy_dir}")

    train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)

    train_cfg = TrainDatasetConfig(
        scale=args.scale,
        patch_size=args.patch_size,
        augment=True,
        augment_profile=args.augment_profile,
        ood_prob=args.ood_prob,
        ood_max_transforms=args.ood_max_transforms,
        ood_clip_min=args.ood_clip_min,
        ood_clip_max=args.ood_clip_max,
    )
    val_cfg = TrainDatasetConfig(scale=args.scale, patch_size=args.patch_size, augment=False, augment_profile="basic")

    train_ds = NpyPairDataset(train_pairs, cfg=train_cfg, training=True)
    val_ds = NpyPairDataset(val_pairs, val_cfg, training=False)

    train_persistent_workers = bool(args.num_workers > 0 and not args.augment_curriculum)
    if is_main_process(rank) and args.augment_curriculum and args.num_workers > 0:
        print("[Augment] Curriculum enabled: disabling persistent train workers for epoch-wise policy updates.")

    train_loader, train_sampler = make_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        device=device,
        persistent_workers=train_persistent_workers,
    )

    val_loader, val_sampler = make_loader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        num_workers=max(1, args.num_workers // 2),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_last=False,
        device=device,
        persistent_workers=(args.num_workers > 0),
    )

    model_cfg = make_nafnet_config(
        preset=args.nafnet_preset,
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        scale=args.scale,
        official_repo=args.official_repo,
        upsample_mode=args.upsample_mode,
    )
    model = NAFNetH2SR(model_cfg).to(device)

    resume_path = resolve_resume_path(args, out_dir)
    pretrained_report = None
    if resume_path is None and args.pretrained:
        pretrained_report = load_nafnet_pretrained(model, args.pretrained, strict=args.strict_pretrained)
        if is_main_process(rank):
            print("Loaded official pretrained:", json.dumps(pretrained_report, indent=2))

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(args.staged_freeze),
        )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.optim_beta1, args.optim_beta2),
    )

    steps_per_epoch = max(len(train_loader), 1)
    target_total_iters = int(args.total_iters) if args.total_iters > 0 else int(args.epochs * steps_per_epoch)
    if target_total_iters <= 0:
        raise ValueError("target_total_iters must be > 0")

    warmup_iters = max(int(args.lr_warmup_epochs * steps_per_epoch), 0)
    warmup_iters = min(warmup_iters, max(target_total_iters - 1, 0))

    if warmup_iters > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=args.lr_warmup_start_factor, end_factor=1.0, total_iters=warmup_iters),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(target_total_iters - warmup_iters, 1),
                    eta_min=args.lr_eta_min,
                ),
            ],
            milestones=[warmup_iters],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(target_total_iters, 1), eta_min=args.lr_eta_min)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(args.amp and device.type == "cuda"))

    lpips_net = None
    if args.lambda_lpips_max > 0:
        msg = "lambda_lpips_max > 0 but lpips is not available. Install with: pip install lpips"
        if not HAS_LPIPS:
            if args.require_lpips:
                raise RuntimeError(msg)
            if is_main_process(rank):
                print(f"[Warning] {msg}. LPIPS term will be zero.")
        else:
            lpips_net = build_lpips_model(net=args.lpips_net, device=device)
            if lpips_net is None and args.require_lpips:
                raise RuntimeError(msg)
            if is_main_process(rank):
                print(f"LPIPS objective enabled with net='{args.lpips_net}'.")
    lpips_enabled = lpips_net is not None and args.lambda_lpips_max > 0

    loss_cfg = LossConfig(
        scale=args.scale,
        h2_a=args.h2_a,
        h2_b=args.h2_b,
        student_t_nu=args.student_t_nu,
        lambda_psnr=args.lambda_psnr,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        lambda_charbonnier=args.lambda_charbonnier,
        lambda_fft=args.lambda_fft,
        lambda_ssim=args.lambda_ssim,
        lambda_dc=args.lambda_dc,
        ssim_contribute_to_loss=args.ssim_contribute_to_loss,
        lambda_lpips=0.0,
        pixel_loss_type=args.pixel_loss_type,
        charbonnier_eps=args.charbonnier_eps,
        lambda_edge=args.lambda_edge,
    )

    dc_schedule_mode = args.dc_schedule
    dc_sched = None
    if dc_schedule_mode == "adaptive":
        dc_sched = DCScheduler(
            start=args.dc_lambda_start,
            step=args.dc_lambda_step,
            cap=args.dc_lambda_cap,
            patience=args.dc_patience,
            min_delta=args.dc_min_delta,
        )

    start_epoch = 0
    global_step = 0
    best_psnr = -1e9
    best_lpips = float("inf")
    lpips_gate_epoch: int | None = None
    prev_val_psnr: float | None = None

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu")
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            if is_main_process(rank):
                print(f"[Warning] Scheduler state could not be restored ({exc}); continuing fresh scheduler state.")
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", start_epoch * steps_per_epoch))
        best_psnr = float(ckpt.get("best_psnr", best_psnr))
        best_lpips = float(ckpt.get("best_lpips", best_lpips))
        if dc_sched is not None:
            dc_sched.load_state_dict(ckpt.get("dc_scheduler", {}))
        lpips_gate_epoch = ckpt.get("lpips_gate_epoch", lpips_gate_epoch)
        prev_val_psnr = ckpt.get("prev_val_psnr", prev_val_psnr)
        if is_main_process(rank):
            print(f"Resumed from {resume_path} at epoch {start_epoch}")

    if is_main_process(rank):
        with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    barrier(distributed)

    max_epochs = max(args.epochs, int(math.ceil(float(target_total_iters) / float(steps_per_epoch))))

    if is_main_process(rank):
        print(
            "Training schedule:",
            json.dumps(
                {
                    "target_total_iters": target_total_iters,
                    "steps_per_epoch": steps_per_epoch,
                    "warmup_iters": warmup_iters,
                    "max_epochs": max_epochs,
                    "finetune_preset": args.finetune_preset,
                    "stage1_end_epoch": args.stage1_end_epoch,
                    "stage2_end_epoch": args.stage2_end_epoch,
                    "augment_profile": args.augment_profile,
                    "augment_curriculum": args.augment_curriculum,
                    "ood_prob": args.ood_prob,
                    "ood_max_transforms": args.ood_max_transforms,
                    "pixel_loss_type": args.pixel_loss_type,
                    "optimizer_betas": [args.optim_beta1, args.optim_beta2],
                    "lr_eta_min": args.lr_eta_min,
                    "lambda_psnr": args.lambda_psnr,
                    "lambda_l1": args.lambda_l1,
                    "lambda_l2": args.lambda_l2,
                    "lambda_charbonnier": args.lambda_charbonnier,
                    "lambda_fft": args.lambda_fft,
                },
                indent=2,
            ),
        )

    freeze_phase_current: str | None = None
    freeze_trainable = 0
    freeze_total = 0

    for epoch in range(start_epoch, max_epochs):
        if global_step >= target_total_iters:
            break

        epoch_t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        aug_policy = effective_augment_policy(args, epoch=epoch, max_epochs=max_epochs)
        train_ds.set_ood_policy(
            profile=str(aug_policy["profile"]),
            prob=float(aug_policy["ood_prob"]),
            max_transforms=int(aug_policy["ood_max_transforms"]),
        )

        freeze_phase = _freeze_phase_for_epoch(args, epoch)
        if freeze_phase != freeze_phase_current:
            freeze_trainable, freeze_total = _apply_freeze_phase(model, args, freeze_phase)
            freeze_phase_current = freeze_phase
            if is_main_process(rank):
                print(
                    f"[Freeze] phase={freeze_phase_current} "
                    f"trainable_params={freeze_trainable}/{freeze_total}"
                )

        model.train()
        running = {
            "loss_total": 0.0,
            "loss_recon": 0.0,
            "loss_psnr": 0.0,
            "loss_l1": 0.0,
            "loss_l2": 0.0,
            "loss_charbonnier": 0.0,
            "loss_fft": 0.0,
            "loss_ssim": 0.0,
            "loss_dc": 0.0,
            "loss_lpips": 0.0,
            "loss_edge": 0.0,
        }
        num_steps = 0
        skipped_nonfinite = 0

        iterator = tqdm(train_loader, disable=not is_main_process(rank), desc=f"Epoch {epoch+1}/{max_epochs}")

        h2_jitter_effective = effective_h2_jitter(args, epoch)
        stage_weights = effective_stage_weights(args, epoch=epoch, max_epochs=max_epochs)

        lam_psnr_effective = stage_weights["lambda_psnr"]
        lam_l1_effective = stage_weights["lambda_l1"]
        lam_l2_effective = stage_weights["lambda_l2"]
        lam_charb_effective = stage_weights["lambda_charbonnier"]
        lam_fft_effective = stage_weights["lambda_fft"]

        if args.finetune_preset == "none":
            lam_ssim_effective = effective_lambda_ssim(args, epoch)
            lam_dc_base = stage_weights["lambda_dc"]
        else:
            lam_ssim_effective = stage_weights["lambda_ssim"]
            lam_dc_base = stage_weights["lambda_dc"]

        if lpips_enabled and lpips_gate_epoch is None:
            gate_epoch_ready = epoch >= args.lpips_start_epoch and epoch >= args.lpips_gate_min_epoch
            gate_psnr_ready = prev_val_psnr is not None and prev_val_psnr >= args.lpips_gate_psnr
            if gate_epoch_ready and gate_psnr_ready:
                lpips_gate_epoch = epoch

        lam_lpips_effective = effective_lambda_lpips(args, epoch, lpips_gate_epoch) if lpips_enabled else 0.0

        if args.finetune_preset == "none":
            use_dc = lam_dc_base > 0 and epoch >= args.warmup_epochs
        else:
            use_dc = lam_dc_base > 0

        if use_dc:
            if dc_schedule_mode == "adaptive" and dc_sched is not None:
                lam_dc_effective = dc_sched.lam
            elif dc_schedule_mode == "constant" and args.finetune_preset == "none":
                lam_dc_effective = lam_dc_base
            else:
                if args.finetune_preset == "none":
                    dc_epoch = epoch - args.warmup_epochs
                    ramp_epochs = max(args.dc_ramp_epochs, 1)
                    lam_dc_effective = min(lam_dc_base, lam_dc_base * float(dc_epoch + 1) / float(ramp_epochs))
                else:
                    lam_dc_effective = lam_dc_base
        else:
            lam_dc_effective = 0.0

        for batch in iterator:
            if global_step >= target_total_iters:
                break

            y = batch["noisy"].to(device, non_blocking=True)
            x = batch["gt"].to(device, non_blocking=True)

            if h2_jitter_effective > 0:
                factor_a = 1.0 + random.uniform(-h2_jitter_effective, h2_jitter_effective)
                factor_b = 1.0 + random.uniform(-h2_jitter_effective, h2_jitter_effective)
                batch_loss_cfg = replace(
                    loss_cfg,
                    h2_a=max(loss_cfg.h2_a * factor_a, 1e-10),
                    h2_b=max(loss_cfg.h2_b * factor_b, 1e-10),
                    lambda_psnr=lam_psnr_effective,
                    lambda_l1=lam_l1_effective,
                    lambda_l2=lam_l2_effective,
                    lambda_charbonnier=lam_charb_effective,
                    lambda_fft=lam_fft_effective,
                    lambda_ssim=lam_ssim_effective,
                    lambda_dc=lam_dc_effective,
                    lambda_lpips=lam_lpips_effective,
                )
            else:
                batch_loss_cfg = replace(
                    loss_cfg,
                    lambda_psnr=lam_psnr_effective,
                    lambda_l1=lam_l1_effective,
                    lambda_l2=lam_l2_effective,
                    lambda_charbonnier=lam_charb_effective,
                    lambda_fft=lam_fft_effective,
                    lambda_ssim=lam_ssim_effective,
                    lambda_dc=lam_dc_effective,
                    lambda_lpips=lam_lpips_effective,
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                pred = model(y)

                if not _all_ranks_finite(pred, distributed=distributed):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    if is_main_process(rank) and skipped_nonfinite <= 3:
                        print(
                            f"[Warning] Non-finite prediction detected at epoch={epoch + 1}, "
                            f"step={num_steps + 1}. Skipping batch."
                        )
                    continue

                if (
                    use_dc
                    and args.dc_debug_first_epoch
                    and epoch == args.warmup_epochs
                    and num_steps == 0
                    and is_main_process(rank)
                ):
                    with torch.no_grad():
                        mu_dbg, q_dbg = forward_consistency_h2(pred, scale=batch_loss_cfg.scale)
                        v_dbg = heteroscedastic_variance_h2(q_dbg, h2_a=batch_loss_cfg.h2_a, h2_b=batch_loss_cfg.h2_b)
                        resid_abs = (y - mu_dbg).abs()
                        print(
                            "[DC debug] "
                            f"epoch={epoch + 1} "
                            f"lam_dc={lam_dc_effective:.5f} "
                            f"v[min,max,mean]=({v_dbg.min().item():.5f},{v_dbg.max().item():.5f},{v_dbg.mean().item():.5f}) "
                            f"|resid|[min,max,mean]=({resid_abs.min().item():.5f},{resid_abs.max().item():.5f},{resid_abs.mean().item():.5f})"
                        )

                loss, logs = combined_restoration_loss(
                    x_hat=pred,
                    x_gt=x,
                    y_lr=y,
                    cfg=batch_loss_cfg,
                    use_dc=use_dc,
                    lpips_net=lpips_net,
                )

            if not _all_ranks_finite(loss, distributed=distributed):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if is_main_process(rank) and skipped_nonfinite <= 3:
                    print(
                        f"[Warning] Non-finite loss detected at epoch={epoch + 1}, "
                        f"step={num_steps + 1}. Skipping batch."
                    )
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step > 1:
                scheduler.step()

            num_steps += 1
            for k in running:
                running[k] += logs[k]

            if is_main_process(rank):
                iterator.set_postfix(
                    loss=f"{logs['loss_total']:.4f}",
                    recon=f"{logs['loss_recon']:.4f}",
                    psnr=f"{logs['loss_psnr']:.4f}",
                    l1=f"{logs['loss_l1']:.4f}",
                    l2=f"{logs['loss_l2']:.4f}",
                    charb=f"{logs['loss_charbonnier']:.4f}",
                    fft=f"{logs['loss_fft']:.4f}",
                    dc=f"{logs['loss_dc']:.4f}",
                    ssim=f"{logs['loss_ssim']:.4f}",
                    edge=f"{logs['loss_edge']:.4f}",
                    lpips=f"{logs['loss_lpips']:.4f}",
                    lam_psnr=f"{lam_psnr_effective:.3f}",
                    lam_l1=f"{lam_l1_effective:.3f}",
                    lam_l2=f"{lam_l2_effective:.3f}",
                    lam_charb=f"{lam_charb_effective:.3f}",
                    lam_fft=f"{lam_fft_effective:.3f}",
                    lam_dc=f"{lam_dc_effective:.4f}",
                    lam_ssim=f"{lam_ssim_effective:.4f}",
                    lam_lpips=f"{lam_lpips_effective:.4f}",
                    h2_jit=f"{h2_jitter_effective:.4f}",
                    dc_on=str(use_dc),
                    it=f"{global_step}/{target_total_iters}",
                )

        if num_steps == 0:
            if global_step >= target_total_iters:
                break
            raise RuntimeError(
                "All batches in this epoch were skipped due to non-finite tensors. "
                "Try lowering LR and disabling AMP (amp=false)."
            )

        epoch_metrics = {k: running[k] / max(num_steps, 1) for k in running}
        epoch_metrics = reduce_dict(epoch_metrics, device=device, distributed=distributed)

        val_metrics = evaluate(model, val_loader, device=device, distributed=distributed, lpips_net=lpips_net)
        prev_val_psnr = float(val_metrics.get("val_psnr", 0.0))

        lam_dc_next = lam_dc_effective
        if use_dc and dc_schedule_mode == "adaptive" and dc_sched is not None and val_metrics:
            lam_dc_next = dc_sched.update(float(val_metrics.get("val_psnr", 0.0)))

        if is_main_process(rank):
            elapsed = time.time() - epoch_t0
            msg = {
                "epoch": epoch,
                "global_step": global_step,
                "target_total_iters": target_total_iters,
                "elapsed_sec": round(elapsed, 2),
                "train": epoch_metrics,
                "val": val_metrics,
                "use_dc": use_dc,
                "dc_schedule": dc_schedule_mode,
                "finetune_preset": args.finetune_preset,
                "lam_psnr": round(float(lam_psnr_effective), 6),
                "lam_l1": round(float(lam_l1_effective), 6),
                "lam_l2": round(float(lam_l2_effective), 6),
                "lam_charbonnier": round(float(lam_charb_effective), 6),
                "lam_fft": round(float(lam_fft_effective), 6),
                "lam_dc": round(float(lam_dc_effective), 6),
                "lam_dc_next": round(float(lam_dc_next), 6),
                "lam_ssim": round(float(lam_ssim_effective), 6),
                "lam_lpips": round(float(lam_lpips_effective), 6),
                "augment_profile": str(aug_policy["profile"]),
                "ood_prob": round(float(aug_policy["ood_prob"]), 4),
                "ood_max_transforms": int(aug_policy["ood_max_transforms"]),
                "lpips_gate_epoch": lpips_gate_epoch,
                "freeze_phase": freeze_phase_current,
                "trainable_params": int(freeze_trainable),
                "total_params": int(freeze_total),
                "skipped_nonfinite": int(skipped_nonfinite),
                "h2_jitter": round(float(h2_jitter_effective), 6),
            }
            print(json.dumps(msg))

            out_dir.mkdir(parents=True, exist_ok=True)

            save_payload = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "target_total_iters": target_total_iters,
                "best_psnr": best_psnr,
                "best_lpips": best_lpips,
                "dc_schedule_mode": dc_schedule_mode,
                "dc_scheduler": dc_sched.state_dict() if dc_sched is not None else None,
                "lpips_gate_epoch": lpips_gate_epoch,
                "prev_val_psnr": prev_val_psnr,
                "pretrained_report": pretrained_report,
                "args": vars(args),
            }

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(save_payload, out_dir / f"checkpoint_epoch_{epoch+1:03d}.pt")

            cur_psnr = float(val_metrics["val_psnr"])
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                save_payload["best_psnr"] = best_psnr
                torch.save(save_payload, out_dir / "best.pt")
                if args.export_best_infer:
                    torch.save(build_infer_export_payload(save_payload), out_dir / "best_infer.pt")

            cur_lpips = float(val_metrics.get("val_lpips", float("inf")))
            if lpips_enabled and np.isfinite(cur_lpips) and cur_lpips < best_lpips:
                best_lpips = cur_lpips
                save_payload["best_lpips"] = best_lpips
                torch.save(save_payload, out_dir / "best_lpips.pt")
                if args.export_best_infer:
                    torch.save(build_infer_export_payload(save_payload), out_dir / "best_lpips_infer.pt")

            torch.save(save_payload, out_dir / "latest.pt")
            if args.export_latest_infer:
                torch.save(build_infer_export_payload(save_payload), out_dir / "latest_infer.pt")

    if is_main_process(rank) and args.export_best_infer:
        best_infer = out_dir / "best_infer.pt"
        latest_infer = out_dir / "latest_infer.pt"
        if not best_infer.exists() and latest_infer.exists():
            shutil.copyfile(latest_infer, best_infer)

    barrier(distributed)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
