"""Configuration loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:  # pragma: no cover - only executed on Python < 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(slots=True)
class ConfigLoader:
    """Load configuration from JSON or TOML files."""

    base_path: Path

    @classmethod
    def from_cwd(cls) -> "ConfigLoader":
        return cls(base_path=Path.cwd())

    def load(self, location: str | Path) -> dict[str, Any]:
        path = (self.base_path / Path(location)).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(path.read_text())
        if suffix == ".toml":
            return tomllib.loads(path.read_text())  # type: ignore[arg-type]

        raise ValueError(
            "Unsupported configuration format. Expected a JSON or TOML file."
        )


def merge_dicts(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge multiple mappings into a new dictionary."""

    merged: dict[str, Any] = {}
    for config in configs:
        _merge_into(merged, config)
    return merged


def _merge_into(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], Mapping)
            and isinstance(value, Mapping)
        ):
            nested: MutableMapping[str, Any] = target[key]  # type: ignore[assignment]
            _merge_into(nested, value)
        else:
            target[key] = value
