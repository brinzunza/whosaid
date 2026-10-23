"""Refuse to start more work when the disk is nearly full (macOS needs free space for swap; a full disk can wedge the
whole machine, not just this project). Threshold: env WHOSAID_MIN_FREE_GB (default 2.0)."""
from __future__ import annotations

import os
import shutil


class LowDisk(RuntimeError):
    pass


def free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 2**30


def ensure_free(what: str = "work", min_gb: float | None = None, path: str = ".") -> float:
    """Raise LowDisk if free space is below the threshold; return the free GiB otherwise."""
    need = float(os.environ.get("WHOSAID_MIN_FREE_GB", 2.0)) if min_gb is None else min_gb
    free = free_gb(path)
    if free < need:
        raise LowDisk(f"only {free:.1f} GiB free (< {need:.1f}); stopping before {what}. Free space "
                      f"or lower WHOSAID_MIN_FREE_GB.")
    return free


def _du(path) -> int:
    from pathlib import Path

    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def report(root: str = ".") -> str:
    from pathlib import Path

    rows = []
    for d in ["data/single", "data/lib", "data/synth", "data/asd", "data/bench", "data/decide", "out", "models", ".venv"]:
        if (Path(root) / d).exists():
            rows.append((_du(Path(root) / d) / 2**30, d))
    lines = [f"free on this volume: {free_gb(root):.1f} GiB (guard stops jobs below {os.environ.get('WHOSAID_MIN_FREE_GB', 2.0)} GiB)", ""]
    lines += [f"  {g:6.2f} GiB  {d}" for g, d in sorted(rows, reverse=True)]
    lines += ["", "regenerable: data/synth (deterministic by seed), data/asd (whosaid asd build), data/decide, data/bench/*/ work dirs",
              "  (the last cost minutes of compute per scene to recreate)"]
    return "\n".join(lines)
