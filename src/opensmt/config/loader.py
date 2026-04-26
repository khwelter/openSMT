from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    if isinstance(base, list) and isinstance(override, list):
        return [*base, *override]

    return override


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_includes(node: Any, base_dir: Path) -> Any:
    if isinstance(node, dict):
        include_value = node.get("$include")
        if include_value is not None:
            includes = include_value if isinstance(include_value, list) else [include_value]
            merged: Any = {}
            for item in includes:
                include_path = (base_dir / str(item)).resolve()
                included = _resolve_includes(_load_json(include_path), include_path.parent)
                merged = deep_merge(merged, included)

            remaining = {k: v for k, v in node.items() if k != "$include"}
            merged = deep_merge(merged, _resolve_includes(remaining, base_dir))
            return merged

        return {k: _resolve_includes(v, base_dir) for k, v in node.items()}

    if isinstance(node, list):
        return [_resolve_includes(item, base_dir) for item in node]

    return node


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).resolve()
    data = _load_json(cfg_path)
    resolved = _resolve_includes(data, cfg_path.parent)
    if not isinstance(resolved, dict):
        raise ValueError("Root configuration must be a JSON object")
    return resolved
