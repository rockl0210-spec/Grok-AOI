"""Config loading. The YAML file is the single source of truth: every tunable lives there.

`Cfg` is a thin read-only wrapper that raises a clear error when a key is missing
(instead of silently falling back to a hidden default).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


class Cfg(Mapping):
    def __init__(self, data: dict, path: str = ""):
        self._d = data
        self._path = path

    def __getitem__(self, key: str) -> Any:
        if key not in self._d:
            raise ConfigError(f"config is missing key '{self._path + key}'")
        v = self._d[key]
        if isinstance(v, dict):
            return Cfg(v, f"{self._path}{key}.")
        return v

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def choice(self, key: str, allowed: tuple[str, ...]) -> str:
        v = self[key]
        if v not in allowed:
            raise ConfigError(f"'{self._path}{key}' must be one of {allowed}, got {v!r}")
        return v

    def to_dict(self) -> dict:
        return self._d


def load_config(path: str | Path) -> Cfg:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return Cfg(data)
