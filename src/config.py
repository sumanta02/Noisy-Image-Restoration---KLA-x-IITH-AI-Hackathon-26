from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config_file(path: str) -> dict[str, Any]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(p)
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Config file must be .yaml/.yml or .json")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a dictionary")
    return data


def _collect_leaf_keys(obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                _collect_leaf_keys(value, out)
            else:
                out[str(key)] = value


def apply_config_defaults(parser: argparse.ArgumentParser, config_data: dict[str, Any]) -> dict[str, Any]:
    if not config_data:
        return {}

    flat: dict[str, Any] = {}
    _collect_leaf_keys(config_data, flat)

    known_dests = {action.dest for action in parser._actions}
    to_apply = {k: v for k, v in flat.items() if k in known_dests}
    unknown = {k: v for k, v in flat.items() if k not in known_dests}

    if to_apply:
        parser.set_defaults(**to_apply)

    return unknown
