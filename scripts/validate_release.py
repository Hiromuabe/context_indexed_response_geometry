"""Static checks for accidental private or generated content in the release."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("/Users/", "/workspace/", "BEGIN PRIVATE KEY", "hf_")
BAD_SUFFIXES = (".pyc", ".npy", ".npz", ".pt", ".pth", ".safetensors")

errors: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    relative = path.relative_to(ROOT)
    # This validator necessarily contains the literal signatures it searches
    # for, so it must not scan its own source.
    if relative == Path("scripts/validate_release.py"):
        continue
    if path.name == ".DS_Store":
        errors.append(f"platform metadata: {relative}")
        continue
    if "__pycache__" in path.parts or path.suffix in BAD_SUFFIXES:
        errors.append(f"generated/binary artifact: {relative}")
        continue
    if path.suffix in {".py", ".md", ".yaml", ".json", ".toml", ".txt"}:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN:
            if marker in text:
                errors.append(f"private marker {marker!r}: {relative}")
        if path.suffix in {".yaml", ".json"}:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON-compatible config {relative}: {exc}")

if errors:
    raise SystemExit("Release validation failed:\n- " + "\n- ".join(errors))
print(f"Release validation passed: {ROOT}")
