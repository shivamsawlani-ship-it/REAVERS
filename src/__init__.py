"""
CAVShield – src package.

Provides:
    AttrDict   – dict subclass with recursive attribute access.
    load_config – loads config/config.yaml, deep-merges overrides,
                  resolves paths to absolute directories and creates them.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class AttrDict(dict):
    """A dictionary subclass that supports attribute-style access.

    Nested plain dicts are automatically converted to AttrDict on access,
    so ``cfg.simulation.world_hz`` works as expected.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a deep copy of *base*.

    - Dict values are merged recursively.
    - All other values in *overrides* replace the corresponding *base* value.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Repo root is two levels up from this file: src/__init__.py -> src -> cavshield
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "config.yaml"


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AttrDict:
    """Load the YAML configuration and return it as an :class:`AttrDict`.

    Parameters
    ----------
    path : str or None
        Path to the YAML config file.  Defaults to ``config/config.yaml``
        relative to the repository root.
    overrides : dict or None
        A (possibly nested) dict whose values are deep-merged on top of the
        file-based config.

    Returns
    -------
    AttrDict
        The fully resolved configuration.  The ``paths`` section is resolved
        to absolute directories relative to the repo root, and every listed
        directory is created if it does not already exist.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # Resolve the paths section to absolute directories and create them.
    paths = cfg.get("paths", {})
    resolved_paths: Dict[str, str] = {}
    for key, rel in paths.items():
        abs_dir = (_REPO_ROOT / rel).resolve()
        abs_dir.mkdir(parents=True, exist_ok=True)
        resolved_paths[key] = str(abs_dir)
    cfg["paths"] = resolved_paths

    return AttrDict(cfg)

