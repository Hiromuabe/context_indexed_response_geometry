"""Reject identifying, generated, secret, or out-of-tree release content."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_MARKERS = (
    "/Users/",
    "/home/",
    "/workspace/",
    "/root/",
    "C:\\Users\\",
    "h_abe",
    "virach",
    "black2026",
    "5996e1cfe848",
    "BEGIN PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
)
SECRET_PATTERNS = {
    "email address": re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
BAD_SUFFIXES = (".pyc", ".npy", ".npz", ".pt", ".pth", ".safetensors")
TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh", ".svg", ".csv")
IGNORED_GENERATED_DIRECTORIES = {
    ".venv", "artifacts", "build", "data", "dist", "logs", "models", "reports", "results",
}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


errors: list[str] = []
for path in ROOT.rglob("*"):
    relative = path.relative_to(ROOT)
    if (
        any(part in IGNORED_GENERATED_DIRECTORIES for part in relative.parts)
        and relative != Path("data/README.md")
    ):
        continue
    relative_text = relative.as_posix()
    if any(marker.lower() in relative_text.lower() for marker in FORBIDDEN_MARKERS):
        errors.append(f"identifying path name: {relative}")
    if path.is_symlink():
        if not _is_within(path, ROOT):
            errors.append(f"symlink escapes release root: {relative} -> {path.resolve()}")
        continue
    if not path.is_file():
        continue
    if ".git" in path.parts:
        errors.append(f"repository history or Git metadata: {relative}")
        continue
    if path.name == ".DS_Store":
        errors.append(f"platform metadata: {relative}")
        continue
    if "__pycache__" in path.parts or path.suffix in BAD_SUFFIXES:
        errors.append(f"generated or binary artifact: {relative}")
        continue
    if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
        continue
    text = path.read_text(encoding="utf-8")
    if relative != Path("scripts/validate_release.py"):
        for marker in FORBIDDEN_MARKERS:
            if marker.lower() in text.lower():
                errors.append(f"private marker {marker!r}: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {relative}")
    if path.suffix in {".yaml", ".yml", ".json"}:
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON-compatible configuration {relative}: {exc}")

if errors:
    raise SystemExit("Release validation failed:\n- " + "\n- ".join(sorted(set(errors))))

file_count = sum(
    path.is_file()
    for path in ROOT.rglob("*")
    if ".git" not in path.parts
    and not any(part in IGNORED_GENERATED_DIRECTORIES for part in path.relative_to(ROOT).parts)
)
print(f"Release validation passed: {file_count} files checked")
