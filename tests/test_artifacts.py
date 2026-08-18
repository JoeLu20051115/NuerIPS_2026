from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from pi05_libero_repro.artifacts import build_manifest, verify_manifest, write_manifest


def test_manifest_is_sorted_and_detects_tamper(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "b.bin").write_bytes(b"bb")
    (root / "a.bin").write_bytes(b"a")
    manifest = tmp_path / "manifest.json"

    write_manifest(root, manifest)

    assert [entry.path for entry in build_manifest(root)] == ["a.bin", "b.bin"]
    assert verify_manifest(root, manifest) == []
    (root / "a.bin").write_bytes(b"changed")
    assert verify_manifest(root, manifest) == ["hash/size mismatch: a.bin"]


def test_manifest_detects_missing_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "expected").write_bytes(b"x")
    manifest = tmp_path / "manifest.json"
    write_manifest(root, manifest)
    (root / "expected").unlink()
    (root / "extra").write_bytes(b"y")

    assert verify_manifest(root, manifest) == ["missing: expected", "unexpected: extra"]


def test_manifest_inside_root_excludes_itself(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "payload").write_bytes(b"payload")
    manifest = root / "manifest.json"

    write_manifest(root, manifest)

    data = json.loads(manifest.read_text())
    assert [entry["path"] for entry in data["files"]] == ["payload"]
    assert verify_manifest(root, manifest) == []


def test_manifest_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    (root / "link").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink not allowed: link"):
        build_manifest(root)


def test_cli_returns_nonzero_for_tampered_payload(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    script = Path(__file__).parents[1] / "scripts" / "verify_artifacts.py"

    created = subprocess.run(
        [sys.executable, str(script), "create", str(root), str(manifest)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr

    payload.write_bytes(b"tampered")
    verified = subprocess.run(
        [sys.executable, str(script), "verify", str(root), str(manifest)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 1
    assert verified.stderr == "hash/size mismatch: payload\n"
