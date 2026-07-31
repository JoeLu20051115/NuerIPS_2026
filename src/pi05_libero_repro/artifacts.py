from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: str
    size: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_exclusion(root: Path, path: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return None


def _build_manifest(root: Path, excluded: frozenset[str]) -> list[ManifestEntry]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")

    entries: list[ManifestEntry] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink not allowed: {relative}")
        if path.is_file():
            entries.append(ManifestEntry(relative, path.stat().st_size, _sha256(path)))
    return entries


def build_manifest(root: Path) -> list[ManifestEntry]:
    return _build_manifest(Path(root), frozenset())


def write_manifest(root: Path, output: Path) -> None:
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if output.is_symlink():
        raise ValueError(f"manifest output is a symlink: {output}")
    excluded = _relative_exclusion(root, output)
    entries = _build_manifest(root, frozenset([excluded]) if excluded else frozenset())
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "files": [asdict(entry) for entry in entries]}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_manifest(root: Path, manifest_path: Path) -> list[str]:
    root = Path(root).resolve(strict=True)
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError(f"invalid manifest format: {manifest_path}")

    expected_entries = [ManifestEntry(**entry) for entry in payload["files"]]
    expected = {entry.path: entry for entry in expected_entries}
    if len(expected) != len(expected_entries):
        raise ValueError(f"duplicate paths in manifest: {manifest_path}")

    excluded = _relative_exclusion(root, manifest_path)
    actual_entries = _build_manifest(root, frozenset([excluded]) if excluded else frozenset())
    actual = {entry.path: entry for entry in actual_entries}

    errors = [f"missing: {path}" for path in sorted(expected.keys() - actual.keys())]
    errors.extend(
        f"hash/size mismatch: {path}"
        for path in sorted(expected.keys() & actual.keys())
        if expected[path] != actual[path]
    )
    errors.extend(f"unexpected: {path}" for path in sorted(actual.keys() - expected.keys()))
    return errors
