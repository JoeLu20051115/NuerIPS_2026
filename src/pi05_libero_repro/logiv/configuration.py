from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_extended_json(
    path: Path | str,
    *,
    _parents: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Load JSON with an optional task-scoped ``extends`` overlay."""

    resolved_path = Path(path).resolve()
    if resolved_path in _parents:
        raise ValueError(f"configuration extends cycle at {resolved_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    parent_value = payload.pop("extends", None)
    if parent_value is None:
        return payload
    if not isinstance(parent_value, str) or not parent_value:
        raise ValueError("configuration extends must be a nonempty path")
    parent_path = Path(parent_value)
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path
    parent = load_extended_json(
        parent_path,
        _parents=_parents | {resolved_path},
    )
    overlay_tasks = payload.pop("tasks", None)
    merged = dict(parent)
    for key, value in payload.items():
        parent_value = parent.get(key)
        if isinstance(parent_value, dict) and isinstance(value, dict):
            merged[key] = {**parent_value, **value}
        else:
            merged[key] = value
    if overlay_tasks is None:
        return merged
    if not isinstance(overlay_tasks, list) or not all(
        isinstance(item, dict) and isinstance(item.get("task_id"), int)
        for item in overlay_tasks
    ):
        raise ValueError("configuration overlay tasks must have integer task_id")
    parent_tasks = parent.get("tasks")
    if not isinstance(parent_tasks, list) or not all(
        isinstance(item, dict) and isinstance(item.get("task_id"), int)
        for item in parent_tasks
    ):
        raise ValueError("extended configuration parent has invalid tasks")
    by_id = {int(item["task_id"]): dict(item) for item in parent_tasks}
    for patch in overlay_tasks:
        task_id = int(patch["task_id"])
        if task_id not in by_id:
            raise ValueError(f"configuration overlay has unknown task_id {task_id}")
        by_id[task_id] = {**by_id[task_id], **patch}
    merged["tasks"] = [by_id[task_id] for task_id in sorted(by_id)]
    return merged


def resolved_json_sha256(path: Path | str) -> str:
    payload = load_extended_json(path)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
