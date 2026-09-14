from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.model import NAFNetH2SR, make_nafnet_config
from src.config import apply_config_defaults, load_config_file
from src.dataset import NpyNoisyDataset
from src.ddp import (
    barrier,
    cleanup_distributed,
    cuda_environment_summary,
    init_distributed_mode,
    is_main_process,
)
from src.losses import dc_loss_robust, forward_consistency_h2, heteroscedastic_variance_h2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed inference for NAFNet H2 model")
    parser.add_argument("--config", type=str, default="", help="Path to YAML/JSON config file")

    parser.add_argument("--input-dir", type=str, default="../data/test/NoisyLR")
    parser.add_argument("--output-dir", type=str, default="outputs/nafnet_h2/test_predictions")
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--train-output-dir", type=str, default="outputs/nafnet_h2")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-preset", type=str, default="fast", choices=["fast", "balanced", "high", "custom"])

    parser.add_argument("--nafnet-preset", type=str, default=None)
    parser.add_argument("--official-repo", type=str, default=None)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--upsample-mode", type=str, default=None)

    parser.add_argument("--h2-a", type=float, default=0.14160734)
    parser.add_argument("--h2-b", type=float, default=6.3383e-05)
    parser.add_argument("--student-t-nu", type=float, default=3.0)

    parser.add_argument("--refine-steps", type=int, default=None)
    parser.add_argument("--refine-lr", type=float, default=None)
    parser.add_argument("--tta", type=str, default=None, choices=["none", "flip4", "flip8"])

    parser.add_argument("--write-submission-csv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--submission-csv-path", type=str, default="")
    parser.add_argument("--verify-all-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-size", type=int, default=256)
    parser.add_argument("--expected-count", type=int, default=-1)

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


def resolve_inference_quality(args: argparse.Namespace) -> tuple[int, float, str]:
    preset_defaults = {
        "fast": {"refine_steps": 0, "refine_lr": 1e-3, "tta": "none"},
        "balanced": {"refine_steps": 3, "refine_lr": 8e-4, "tta": "flip4"},
        "high": {"refine_steps": 8, "refine_lr": 5e-4, "tta": "flip8"},
        "custom": {"refine_steps": 0, "refine_lr": 1e-3, "tta": "none"},
    }

    base = preset_defaults[args.quality_preset]
    refine_steps = int(args.refine_steps) if args.refine_steps is not None else int(base["refine_steps"])
    refine_lr = float(args.refine_lr) if args.refine_lr is not None else float(base["refine_lr"])
    tta_mode = str(args.tta) if args.tta is not None else str(base["tta"])

    if refine_steps < 0:
        raise ValueError("refine_steps must be >= 0")
    if refine_steps > 0 and refine_lr <= 0:
        raise ValueError("refine_lr must be > 0 when refine_steps > 0")

    return refine_steps, refine_lr, tta_mode


def resolve_checkpoint_path(checkpoint_arg: str, train_output_dir: str) -> Path:
    if checkpoint_arg and checkpoint_arg.lower() != "auto":
        p = Path(checkpoint_arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    base = Path(train_output_dir)
    candidates = [
        base / "best_lpips_infer.pt",
        base / "best_infer.pt",
        base / "best_lpips.pt",
        base / "best.pt",
        base / "latest_infer.pt",
        base / "latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not auto-resolve checkpoint in {base}. Expected one of: "
        "best_lpips_infer.pt, best_infer.pt, best_lpips.pt, best.pt, latest_infer.pt, latest.pt"
    )


def validate_prediction_array(arr: np.ndarray, expected_size: int) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape != (expected_size, expected_size):
        raise ValueError(f"Expected shape ({expected_size}, {expected_size}), got {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Prediction contains NaN or Inf values")
    return arr


def write_submission_csv(submission_dir: Path, csv_path: Path, expected_count: int, expected_size: int) -> int:
    files = sorted([f for f in os.listdir(submission_dir) if f.endswith(".npy")])
    if expected_count >= 0 and len(files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} .npy files in {submission_dir}, found {len(files)}")

    rows = []
    for idx, file_name in enumerate(files, start=1):
        arr = np.load(submission_dir / file_name)
        arr = validate_prediction_array(arr, expected_size=expected_size)

        buffer = BytesIO()
        np.save(buffer, arr)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        rows.append({"id": idx, "npy_base64": encoded})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "npy_base64"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def build_model_from_checkpoint_args(args: argparse.Namespace, ckpt_args: dict, device: torch.device) -> torch.nn.Module:
    nafnet_preset = args.nafnet_preset if args.nafnet_preset is not None else ckpt_args.get("nafnet_preset", "sidd-width64")
    official_repo = args.official_repo if args.official_repo is not None else ckpt_args.get("official_repo", "nafnet/official")
    in_channels = args.in_channels if args.in_channels is not None else ckpt_args.get("in_channels", 1)
    scale = args.scale if args.scale is not None else ckpt_args.get("scale", 2)
    upsample_mode = args.upsample_mode if args.upsample_mode is not None else ckpt_args.get("upsample_mode", "bilinear")

    cfg = make_nafnet_config(
        preset=nafnet_preset,
        in_channels=in_channels,
        out_channels=in_channels,
        scale=scale,
        official_repo=official_repo,
        upsample_mode=upsample_mode,
    )
    model = NAFNetH2SR(cfg).to(device)
    return model


def _apply_ops(x: torch.Tensor, ops: Sequence[str]) -> torch.Tensor:
    out = x
    for op in ops:
        if op == "h":
            out = torch.flip(out, dims=(-1,))
        elif op == "v":
            out = torch.flip(out, dims=(-2,))
        elif op == "t":
            out = out.transpose(-2, -1)
        else:
            raise ValueError(f"Unknown TTA op: {op}")
    return out


def _tta_opsets(mode: str) -> list[tuple[str, ...]]:
    if mode == "none":
        return [tuple()]
    if mode == "flip4":
        return [tuple(), ("h",), ("v",), ("h", "v")]
    if mode == "flip8":
        return [tuple(), ("h",), ("v",), ("h", "v"), ("t",), ("t", "h"), ("t", "v"), ("t", "h", "v")]
    raise ValueError(f"Unknown tta mode: {mode}")


def predict_with_tta(model: torch.nn.Module, y: torch.Tensor, tta_mode: str) -> torch.Tensor:
    opsets = _tta_opsets(tta_mode)
    pred_sum = None
    for ops in opsets:
        y_aug = _apply_ops(y, ops)
        pred_aug = model(y_aug)
        pred = _apply_ops(pred_aug, tuple(reversed(ops)))
        if pred_sum is None:
            pred_sum = pred
        else:
            pred_sum = pred_sum + pred

    if pred_sum is None:
        raise RuntimeError("TTA prediction failed: no augmentations were applied")
    return pred_sum / float(len(opsets))


def refine_with_data_consistency(
    pred_hr: torch.Tensor,
    y_lr: torch.Tensor,
    h2_a: float,
    h2_b: float,
    nu: float,
    scale: int,
    steps: int,
    lr: float,
) -> torch.Tensor:
    if steps <= 0:
        return pred_hr

    x_opt = pred_hr.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x_opt], lr=lr)

    for _ in range(steps):
        mu, q = forward_consistency_h2(x_opt, scale=scale)
        var = heteroscedastic_variance_h2(q, h2_a=h2_a, h2_b=h2_b)
        loss = dc_loss_robust(y_lr, mu, var, nu=nu)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            x_opt.clamp_(0.0, 1.0)

    return x_opt.detach()


def main() -> None:
    args = parse_args()
    refine_steps, refine_lr, tta_mode = resolve_inference_quality(args)

    env = init_distributed_mode()
    rank = env["rank"]
    world_size = env["world_size"]
    distributed = env["distributed"]
    device = env["device"]

    if is_main_process(rank):
        print("CUDA environment:", json.dumps(cuda_environment_summary(), indent=2))
        print(f"Distributed: {distributed}, world_size={world_size}")
        print(
            "Inference quality:",
            json.dumps(
                {
                    "quality_preset": args.quality_preset,
                    "refine_steps": refine_steps,
                    "refine_lr": refine_lr,
                    "tta": tta_mode,
                },
                indent=2,
            ),
        )

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.train_output_dir)
    if is_main_process(rank):
        print(f"Using checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    model = build_model_from_checkpoint_args(args, ckpt_args=ckpt_args, device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    scale = int(args.scale if args.scale is not None else ckpt_args.get("scale", 2))

    if distributed:
        model = DDP(model, device_ids=[env["local_rank"]] if device.type == "cuda" else None)

    dataset = NpyNoisyDataset(args.input_dir)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    iterator = tqdm(loader, disable=not is_main_process(rank), desc="Inference")

    for batch in iterator:
        y = batch["noisy"].to(device, non_blocking=True)
        names = batch["name"]

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                pred = predict_with_tta(model, y, tta_mode=tta_mode)

        if refine_steps > 0:
            pred = refine_with_data_consistency(
                pred_hr=pred,
                y_lr=y,
                h2_a=args.h2_a,
                h2_b=args.h2_b,
                nu=args.student_t_nu,
                scale=scale,
                steps=refine_steps,
                lr=refine_lr,
            )

        pred = pred.clamp(0.0, 1.0)

        for i, name in enumerate(names):
            arr = pred[i].detach().cpu().numpy()
            arr = validate_prediction_array(arr, expected_size=args.expected_size)
            np.save(output_dir / name, arr)

    barrier(distributed)
    if is_main_process(rank):
        produced = len([f for f in os.listdir(output_dir) if f.endswith(".npy")])
        if args.verify_all_inputs and produced != len(dataset):
            raise RuntimeError(f"Expected {len(dataset)} prediction files, found {produced}")

        print(f"Saved {produced} predictions to {output_dir}")

        if args.write_submission_csv:
            csv_path = Path(args.submission_csv_path) if args.submission_csv_path else output_dir.parent / "submission.csv"
            expected_count = args.expected_count if args.expected_count >= 0 else len(dataset)
            rows = write_submission_csv(
                submission_dir=output_dir,
                csv_path=csv_path,
                expected_count=expected_count,
                expected_size=args.expected_size,
            )
            print(f"Wrote submission CSV with {rows} rows to {csv_path}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
