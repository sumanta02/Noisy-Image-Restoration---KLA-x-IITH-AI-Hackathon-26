from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _resolve_state_dict(payload: object) -> tuple[str, dict]:
    if isinstance(payload, dict):
        if "params" in payload and isinstance(payload["params"], dict):
            return "params", payload["params"]
        if "params_ema" in payload and isinstance(payload["params_ema"], dict):
            return "params_ema", payload["params_ema"]
        if "model" in payload and isinstance(payload["model"], dict):
            return "model", payload["model"]
        # Fallback: if it already looks like a state_dict.
        if payload and all(isinstance(k, str) for k in payload.keys()):
            maybe_tensor_values = any(torch.is_tensor(v) for v in payload.values())
            if maybe_tensor_values:
                return "root", payload
    raise ValueError("Could not locate a model state dict in checkpoint payload")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect official NAFNet checkpoint layout")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    source_key, state = _resolve_state_dict(payload)

    keys = list(state.keys())
    print(f"Checkpoint: {ckpt_path}")
    print(f"State source: {source_key}")
    print(f"Parameter tensors: {len(keys)}")

    sample = keys[:20]
    print("Sample keys:")
    for name in sample:
        tensor = state[name]
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else "<non-tensor>"
        print(f"  - {name}: {shape}")


if __name__ == "__main__":
    main()
