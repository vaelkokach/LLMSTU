"""Run provenance and GPU selection for Branch C.

BRANCH_C_PROTOCOL.md section 9 requires every run to record its git commit,
dirty-diff hash, environment hash, dataset/split hash, feature-schema hash,
checkpoint hash, command, seed, host/GPU, wall time and peak memory. This module
is the single implementation of that record, so the fields cannot drift between
the sweep launcher, the evaluator and the dashboard.

Two things here are deliberately paranoid, both because this project has been
bitten by the cheaper version:

*Atomic append.* ``RUNS.jsonl`` is written by up to four concurrent single-GPU
processes. A partial line in an evidence file is worse than a missing one,
because it still parses as "something happened". Records are serialised to a
single line and appended under an advisory lock, and every record carries a
``complete`` sentinel written last.

*GPU selection by measurement, not assumption.* The host exposes eight A100s but
the standing project constraint is that at most four may be occupied, and the box
is shared - at audit time GPUs 0 and 4 already carried other users' processes
while the repo's own launcher defaults to ``0,1,2,3``. Selecting free devices by
querying actual memory use avoids landing a job on top of somebody else's work.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO / "outputs/branch_c/RUNS.jsonl"

# At most four of the eight A100s may be occupied by this project by default. The
# box is shared, and that default exists so a routine run cannot starve another
# user. It is overridable only through an explicit environment variable, never
# from a config file and never silently: raising it is a decision someone has to
# take deliberately, per run, having checked the box is actually free.
#
# Raised to 8 for the 2026-08-10 Branch-C clean reruns on the researcher's explicit
# authorisation, with GPUs 4-7 verified idle (21 MiB, 0%) beforehand.
MAX_CONCURRENT_GPUS = int(os.environ.get("BRANCH_C_MAX_GPUS", "4"))

# A device holding more than this is treated as in use by somebody else.
# Idle A100s in this cluster report ~21 MiB.
GPU_FREE_THRESHOLD_MIB = 512


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(
            cmd, cwd=cwd or REPO, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception:
        return ""


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """Hash a file, or None if it is absent. Never raises on a missing artifact."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_state() -> Dict[str, Any]:
    """Commit, branch and a hash of the uncommitted diff.

    The diff hash matters more than the dirty flag: "dirty" tells you the run is
    unreproducible, the hash tells you *which* unreproducible state it was, and
    lets two runs from the same working tree be recognised as comparable.
    """
    diff = _run(["git", "diff", "HEAD"])
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]) or None,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None,
        "dirty": bool(diff),
        "diff_sha256": sha256_text(diff) if diff else None,
        "untracked_sha256": sha256_text(untracked) if untracked else None,
    }


def environment_state() -> Dict[str, Any]:
    """Interpreter, key library versions and a hash over the installed set.

    The repository has no lockfile at all (FINDINGS section 12.6), so this record
    is currently the *only* description of what an experiment ran against.
    """
    versions: Dict[str, Optional[str]] = {}
    for mod in ("torch", "numpy", "cv2", "transformers", "sklearn", "mediapipe"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None

    cuda: Dict[str, Any] = {}
    try:
        import torch

        cuda = {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception:
        pass

    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "versions": versions,
        "cuda": cuda,
        "pip_freeze_sha256": sha256_text(freeze) if freeze else None,
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }


def gpu_status() -> List[Dict[str, Any]]:
    """Per-device index, name, total and used memory, from nvidia-smi."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return devices


def select_free_gpus(
    n: int = MAX_CONCURRENT_GPUS,
    threshold_mib: int = GPU_FREE_THRESHOLD_MIB,
    exclude: Optional[List[int]] = None,
) -> List[int]:
    """Pick up to ``n`` genuinely idle devices, never more than the project cap.

    Returns the emptiest devices first. Raises if nothing is free rather than
    silently queueing onto a busy card, because sharing a device with another
    user's job corrupts both the timing measurements this protocol reports and,
    potentially, the other user's run.
    """
    if n > MAX_CONCURRENT_GPUS:
        raise ValueError(
            f"requested {n} GPUs; the standing project cap is {MAX_CONCURRENT_GPUS} "
            "because the host is shared"
        )
    exclude = set(exclude or [])
    free = [
        d
        for d in gpu_status()
        if d["memory_used_mib"] <= threshold_mib and d["index"] not in exclude
    ]
    free.sort(key=lambda d: (d["memory_used_mib"], d["index"]))
    if not free:
        raise RuntimeError(
            "no idle GPU found; every visible device is above "
            f"{threshold_mib} MiB. Refusing to share a device with another job."
        )
    return [d["index"] for d in free[:n]]


@dataclass
class RunRecord:
    """One row of outputs/branch_c/RUNS.jsonl."""

    run_id: str
    arm: str
    command: List[str]
    seed: Optional[int] = None
    fold: Optional[int] = None
    stage: str = "train"
    split_manifest_sha256: Optional[str] = None
    feature_schema_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    device: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    wall_clock_s: Optional[float] = None
    peak_memory_mib: Optional[int] = None
    returncode: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    git: Dict[str, Any] = field(default_factory=git_state)
    environment: Dict[str, Any] = field(default_factory=environment_state)
    # Written last, and checked by every reader. A row without it is a run that
    # died mid-write and must not be counted as evidence.
    complete: bool = False


def append_record(record: RunRecord, path: Path = DEFAULT_RUNS) -> None:
    """Append one complete record under an advisory lock.

    The lock makes concurrent single-GPU writers safe; the single-line
    serialisation makes a torn write detectable rather than plausible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record.complete = True
    line = json.dumps(asdict(record), sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_records(path: Path = DEFAULT_RUNS, complete_only: bool = True) -> List[dict]:
    """Read RUNS.jsonl, skipping torn or incomplete rows.

    A malformed line is skipped rather than raising: the file is evidence from
    crashed runs as well as good ones, and one bad row must not make the rest
    unreadable.
    """
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if complete_only and not rec.get("complete"):
            continue
        out.append(rec)
    return out


def atomic_write_json(obj: Any, path: Path) -> None:
    """Write JSON via a temp file and rename, so a partial file never appears valid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def write_sentinel(directory: Path, payload: Optional[dict] = None) -> None:
    """Mark a cache/output directory complete. Absence means 'do not trust this'."""
    atomic_write_json(
        {"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(payload or {})},
        Path(directory) / "_COMPLETE.json",
    )


def is_complete(directory: Path) -> bool:
    return (Path(directory) / "_COMPLETE.json").is_file()


if __name__ == "__main__":
    print("git:", json.dumps(git_state(), indent=2))
    print("gpus:")
    for d in gpu_status():
        print("  ", d)
    try:
        print("selected:", select_free_gpus())
    except RuntimeError as exc:
        print("selection failed:", exc)
