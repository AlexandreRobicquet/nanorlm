"""Contained paths and atomic writes for generated evidence artifacts."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def artifact_path(root: str | Path, *parts: str) -> Path:
    """Reject traversal and symlink components before reading or writing a bundle."""
    base = Path(root).absolute()
    candidate = base.joinpath(*parts)
    if any(Path(part).is_absolute() or '..' in Path(part).parts for part in parts):
        raise ValueError('artifact path must stay inside its output root')
    if base.is_symlink():
        raise ValueError('artifact output root must not be a symlink')
    try:
        relative = candidate.relative_to(base)
        candidate.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError('artifact path escapes its output root') from exc
    current = base
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'artifact path contains a symlink: {current.name}')
    return candidate


def write_text_atomic(path: str | Path, text: str) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ValueError('artifact file must not be a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.write-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
