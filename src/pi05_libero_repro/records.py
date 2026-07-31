from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EpisodeRecord:
    checkpoint: str
    task_id: int
    task_name: str
    episode_idx: int
    init_state_sha256: str
    seed: int
    success: bool
    valid: bool
    steps: int
    inference_requests: int
    wall_seconds: float
    exception: Optional[str]
    first_frame_sha256: str
    action_min: float
    action_max: float
    action_mean: float
    done: bool
    check_success: bool
    video_path: str

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.checkpoint, self.task_id, self.episode_idx)

    @property
    def key_text(self) -> str:
        return f"{self.checkpoint}/{self.task_id}/{self.episode_idx}"


def load_records(path: Path) -> List[EpisodeRecord]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line: {path}:{line_number}")
            records.append(EpisodeRecord(**json.loads(line)))
    return records


def append_record(path: Path, record: EpisodeRecord) -> None:
    errors = validate_records([record], expected_trials=1)
    if errors:
        raise ValueError("; ".join(errors))

    path = Path(path)
    existing = load_records(path)
    if record.key in {item.key for item in existing}:
        raise ValueError(f"duplicate episode: {record.key_text}")

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(asdict(record), allow_nan=False, separators=(",", ":"), sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_records(records: Sequence[EpisodeRecord], expected_trials: int) -> List[str]:
    errors = []
    if len(records) != expected_trials:
        errors.append(f"count mismatch: {len(records)} != {expected_trials}")

    seen = set()
    for record in records:
        if record.key in seen:
            errors.append(f"duplicate episode: {record.key_text}")
        seen.add(record.key)
        if not record.valid:
            errors.append(f"invalid episode: {record.key_text}")
        if record.exception is not None:
            errors.append(f"episode exception: {record.key_text}")
        if record.success != record.done or record.done != record.check_success:
            errors.append(f"predicate mismatch: {record.key_text}")
        if not _SHA256.fullmatch(record.init_state_sha256):
            errors.append(f"invalid init-state hash: {record.key_text}")
        if not _SHA256.fullmatch(record.first_frame_sha256):
            errors.append(f"invalid first-frame hash: {record.key_text}")
        if not all(math.isfinite(value) for value in (record.action_min, record.action_max, record.action_mean)):
            errors.append(f"non-finite action statistics: {record.key_text}")
        if not math.isfinite(record.wall_seconds) or record.wall_seconds < 0:
            errors.append(f"invalid wall time: {record.key_text}")
        if record.steps < 0 or record.inference_requests < 0:
            errors.append(f"invalid counters: {record.key_text}")
    return errors


def wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")

    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
    margin /= denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
