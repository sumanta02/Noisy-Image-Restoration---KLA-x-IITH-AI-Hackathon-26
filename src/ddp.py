from __future__ import annotations

import os
import subprocess
from typing import Dict

import torch
import torch.distributed as dist


def init_distributed_mode() -> dict:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "distributed": distributed,
        "device": device,
    }


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def reduce_dict(metrics: Dict[str, float], device: torch.device, distributed: bool) -> Dict[str, float]:
    if not distributed:
        return metrics

    keys = sorted(metrics.keys())
    vals = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= dist.get_world_size()
    return {k: float(v) for k, v in zip(keys, vals.tolist())}


def cuda_environment_summary() -> dict:
    nvidia_smi_gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.strip().splitlines():
            idx_str, name = [x.strip() for x in line.split(",", 1)]
            nvidia_smi_gpus.append({"index": int(idx_str), "name": name})
    except Exception:
        nvidia_smi_gpus = []

    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "torch_gpu_count": int(torch.cuda.device_count()),
            "torch_gpus": [],
            "nvidia_smi_gpus": nvidia_smi_gpus,
        }

    gpus = []
    for idx in range(torch.cuda.device_count()):
        gpus.append({"index": idx, "name": torch.cuda.get_device_name(idx)})

    return {
        "cuda": True,
        "torch_gpu_count": len(gpus),
        "torch_gpus": gpus,
        "nvidia_smi_gpus": nvidia_smi_gpus,
    }
