"""What the dashboard can be pointed at: uploaded videos and built caches.

Two kinds of thing live here.

**Videos** in ``uploads/`` — recordings the user dropped on the page, plus
anything they put in that directory by hand. A video on its own can be streamed
through the full pipeline, which is slow and cannot switch models mid-run.

**Sessions** in ``sessions/`` — caches built by ``precompute_session`` from one
video. A session can be replayed instantly and switched between models freely,
because everything except the temporal head is already computed.

So an uploaded video is *analysed once* into a session, and used from the
session thereafter. The pairing is by name (``lecture.mp4`` ->
``sessions/lecture``) rather than by a database, so the state of the system is
whatever is on disk — a half-built session is a directory without a
``meta.json``, and is reported as absent rather than as broken.

Upload safety
-------------
The server accepts file writes, so filenames coming from a browser are treated
as hostile: only the basename survives, only an allowlisted set of characters
and extensions is accepted, and the result is confined to ``uploads/`` with a
final containment check. Uploads are also size-capped and streamed to disk
rather than buffered, because a lecture recording is larger than the RAM anyone
wants to spend on it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
UPLOAD_DIR = HERE / "uploads"
SESSION_DIR = HERE / "sessions"

#: Containers OpenCV opens reliably. Not an exhaustive list of what it *can*
#: open — an allowlist is the point.
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024      # 8 GB

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """A filename that cannot escape ``uploads/`` and cannot be a surprise.

    Takes the basename only (so ``../../etc/passwd`` and
    ``C:\\Windows\\x.mp4`` both reduce to a leaf), collapses everything outside
    ``[A-Za-z0-9._-]`` to underscores, strips leading dots so nothing lands as a
    hidden file, and requires an allowlisted video extension.
    """
    leaf = Path(name.replace("\\", "/")).name
    stem, dot, ext = leaf.rpartition(".")
    if not dot:
        raise ValueError("filename has no extension")
    ext = "." + _SAFE.sub("", ext).lower()
    if ext not in VIDEO_EXTS:
        raise ValueError(
            f"{ext!r} is not an accepted video type "
            f"({', '.join(sorted(VIDEO_EXTS))})")
    stem = _SAFE.sub("_", stem).lstrip(".") or "upload"
    return f"{stem[:120]}{ext}"


def upload_path(name: str) -> Path:
    """Resolved destination for an upload, checked to be inside ``uploads/``."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = (UPLOAD_DIR / safe_name(name)).resolve()
    # Belt and braces: safe_name should make this unreachable, but a path that
    # escapes the upload directory must never be written to.
    if UPLOAD_DIR.resolve() not in p.parents:
        raise ValueError("refusing to write outside the upload directory")
    return p


def unique_path(p: Path) -> Path:
    """``lecture.mp4`` -> ``lecture-2.mp4`` when the name is taken.

    Overwriting would silently invalidate any session already built from the
    old file of that name.
    """
    if not p.exists():
        return p
    for i in range(2, 1000):
        cand = p.with_name(f"{p.stem}-{i}{p.suffix}")
        if not cand.exists():
            return cand
    raise ValueError("too many files with that name")


def save_upload(rfile, length: int, name: str) -> Path:
    """Stream ``length`` bytes from ``rfile`` to ``uploads/``.

    Streamed in chunks, not read whole: a lecture recording does not belong in
    memory. A short or oversized body leaves no partial file behind.
    """
    if length <= 0:
        raise ValueError("empty upload")
    if length > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"upload is {length / 1e9:.1f} GB; the cap is "
            f"{MAX_UPLOAD_BYTES / 1e9:.0f} GB")
    dest = unique_path(upload_path(name))
    got = 0
    try:
        with open(dest, "wb") as fh:
            while got < length:
                chunk = rfile.read(min(1 << 20, length - got))
                if not chunk:
                    raise ValueError(
                        f"upload ended early: {got} of {length} bytes")
                fh.write(chunk)
                got += len(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def session_dir_for(video: Path) -> Path:
    return SESSION_DIR / video.stem


def session_meta(d: Path) -> Optional[Dict]:
    """A session's metadata, or None if it is absent or half-built."""
    m = d / "meta.json"
    if not (m.exists() and (d / "features.npz").exists()):
        return None
    try:
        return json.loads(m.read_text())
    except json.JSONDecodeError:
        return None


def list_sources(extra_video: Optional[str] = None,
                 extra_session: Optional[str] = None) -> List[Dict]:
    """Everything the dashboard can be pointed at, sessions first.

    ``extra_video`` / ``extra_session`` are whatever was passed on the command
    line, so a source given with ``--video`` appears in the picker alongside the
    uploaded ones instead of being invisible to the UI.
    """
    out: List[Dict] = []
    seen_sessions = set()

    def add_session(d: Path, origin: str = "session"):
        d = d.resolve()
        if d in seen_sessions:
            return
        meta = session_meta(d)
        if meta is None:
            return
        seen_sessions.add(d)
        out.append({
            "id": f"session:{d}",
            "kind": "session",
            "name": d.name,
            "origin": origin,
            "path": str(d),
            "ready": True,
            "n_frames": meta.get("n_frames"),
            "n_tracks": meta.get("n_tracks"),
            "fps": meta.get("fps"),
            "built_on": meta.get("device"),
            "built_s": meta.get("wall_clock_s"),
            "video": meta.get("video"),
        })

    if SESSION_DIR.exists():
        for d in sorted(SESSION_DIR.iterdir()):
            if d.is_dir():
                add_session(d)
    if extra_session:
        add_session(Path(extra_session), origin="command line")

    videos = []
    if UPLOAD_DIR.exists():
        videos += [p for p in sorted(UPLOAD_DIR.iterdir())
                   if p.suffix.lower() in VIDEO_EXTS]
    if extra_video:
        p = Path(extra_video)
        # A device index or a stream URL is not a file and has no session.
        if p.suffix.lower() in VIDEO_EXTS and p.exists():
            videos.append(p.resolve())

    for v in videos:
        sd = session_dir_for(v)
        out.append({
            "id": f"video:{v.resolve()}",
            "kind": "video",
            "name": v.name,
            "origin": "upload" if v.parent.resolve() == UPLOAD_DIR.resolve()
                      else "command line",
            "path": str(v.resolve()),
            "size_mb": round(v.stat().st_size / 1e6, 1),
            "session": str(sd) if session_meta(sd) else None,
            "ready": session_meta(sd) is not None,
        })
    return out


def parse_id(source_id: str):
    """``"session:/abs/path"`` -> ``("session", Path(...))``."""
    kind, _, path = str(source_id).partition(":")
    if kind not in ("session", "video") or not path:
        raise ValueError(f"malformed source id {source_id!r}")
    return kind, Path(path)


def delete_session(d: Path) -> None:
    d = Path(d).resolve()
    if SESSION_DIR.resolve() not in d.parents:
        raise ValueError("refusing to delete outside the session directory")
    shutil.rmtree(d, ignore_errors=True)
