from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {key!r} is missing")
    return value


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply an overlay without modifying the base configuration."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class TeleopConfig:
    raw: dict[str, Any]
    overlays: tuple[str, ...] = ()

    @classmethod
    def load(
        cls, path: str | Path, overlays: "Sequence[str | Path]" = ()
    ) -> "TeleopConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        applied: list[str] = []
        for overlay_path in overlays:
            with Path(overlay_path).open("r", encoding="utf-8") as stream:
                patch = json.load(stream)
            if not isinstance(patch, dict):
                raise ValueError(f"overlay {overlay_path} must be a JSON object")
            data = merge(data, patch)
            applied.append(Path(overlay_path).name)
        for key in ("network", "controls", "base", "head", "robot"):
            _section(data, key)
        return cls(data, tuple(applied))

    def section(self, key: str) -> dict[str, Any]:
        return _section(self.raw, key)

    @property
    def network(self) -> dict[str, Any]:
        return self.section("network")

    @property
    def controls(self) -> dict[str, Any]:
        return self.section("controls")

    @property
    def base(self) -> dict[str, Any]:
        return self.section("base")

    @property
    def head(self) -> dict[str, Any]:
        return self.section("head")

    @property
    def robot(self) -> dict[str, Any]:
        return self.section("robot")


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.json"
