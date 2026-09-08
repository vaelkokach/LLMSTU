#!/usr/bin/env python
"""Hugging Face Space entrypoint.

Fetches the artifacts the dashboard needs from a private model repo, then starts
`tools/dashboard/server.py` on the port Spaces routes to.

Why weights are fetched at runtime rather than baked into the image:

* `artifacts.lock.json` marks the detector and the LLMSTU corpus **restricted**,
  gated on an unresolved ethics/consent question. Baking them into image layers
  would put them in every copy of the Space and in its build cache, where they
  cannot be revoked. Fetching under a token keeps the access decision in one
  place — the model repo's own permissions.
* Temporal checkpoints are ~50 MB each and there are 13 deployable ones. Layers
  that large make every rebuild slow for no benefit, since they change far less
  often than the code.

Environment (set these as Space secrets/variables, not in the Dockerfile):

    HF_TOKEN         read token for the private artifact repo   (secret)
    ARTIFACT_REPO    e.g. "WaelK/llmstu-dashboard-artifacts"    (variable)
    ARTIFACT_TYPE    "model" (default) or "dataset"             (variable)
    DASHBOARD_MODEL  registry id, default arch/mstcn_556_hp     (variable)
    SESSION          session dir name under sessions/, default 0325
    RUNTIME_CONFIG   detector+features config for live analysis  (variable)
    DEVICE           cuda:0 (default) or cpu                     (variable)
    PERSIST_DIR      durable mount path, if not /data            (variable)
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

HOME = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "7860"))


def _cache_root() -> Path:
    """Prefer Spaces persistent storage, fall back to the ephemeral image.

    With a 1-hour sleep timer the Space cold-starts often, and the artifacts are
    ~5 GB (the detector alone is 4.29 GB). On ephemeral disk that is re-fetched
    on every wake, which dominates time-to-first-frame. /data survives sleeps, so
    the download happens once.

    PERSIST_DIR overrides the location, because Spaces offers more than one kind
    of durable storage and they do not all mount at the same path — a mounted
    bucket in particular may appear somewhere else entirely. Set PERSIST_DIR to
    whatever the Space actually mounts and this follows it.

    Each candidate is probed by _writable(), which both writes and gives up.
    """
    candidates = []
    override = os.environ.get("PERSIST_DIR", "").strip()
    if override:
        candidates.append(Path(override))
    candidates += [Path("/data"), Path("/mnt/data")]

    for cand in candidates:
        if _writable(cand):
            return cand
        print(f"[app] {cand} not usable for the cache — skipping", flush=True)
    return HOME


def _writable(cand: Path, timeout_s: float = 20.0) -> bool:
    """Can we actually write here — answered within a bounded time?

    The probe writes a file rather than calling exists(), because /data is
    present but read-only when persistent storage is off, and an exists() check
    would route the cache somewhere that fails on first write.

    The bound matters just as much. A mounted bucket can wedge: the FUSE call
    blocks and never returns, and because this runs before the first print, the
    Space dies silently after its startup banner with no clue why. That happened.
    A daemon thread lets a wedged mount be abandoned instead of waited on, so the
    app falls back to ephemeral disk and boots.
    """
    ok = []

    def probe():
        try:
            cand.mkdir(parents=True, exist_ok=True)
            p = cand / ".write_test"
            p.write_text("ok")
            p.unlink()
            ok.append(True)
        except Exception:
            pass

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout_s)
    return bool(ok)


CACHE_ROOT = _cache_root()
# Set before huggingface_hub is imported anywhere, or it reads the default.
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / ".cache" / "huggingface"))
print(f"[app] cache root: {CACHE_ROOT} "
      f"({'persistent' if CACHE_ROOT != HOME else 'EPHEMERAL — artifacts re-download on every wake'})",
      flush=True)

# Where server.py expects things, relative to the repo root it is run from.
SESSIONS_DIR = HOME / "tools" / "dashboard" / "sessions"
WORKDIRS = HOME / "LLMDet" / "work_dirs"
#: '../huggingface/...' in the detector config resolves from cwd (== HOME).
HF_MODELS = HOME.parent / "huggingface"


def fetch_artifacts() -> bool:
    """Pull checkpoints + session cache from the private repo. True if usable."""
    repo = os.environ.get("ARTIFACT_REPO", "").strip()
    if not repo:
        print("[app] ARTIFACT_REPO not set — starting in replay-only mode "
              "(cue logs render; no model switching).", flush=True)
        return False

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[app] ARTIFACT_REPO is set but HF_TOKEN is not. A private repo "
              "cannot be read without it; add it as a Space secret.", flush=True)
        return False

    from huggingface_hub import snapshot_download

    def pull(allow, dest, label):
        print(f"[app] fetching {label} -> {dest}", flush=True)
        return Path(snapshot_download(
            repo_id=repo, repo_type=os.environ.get("ARTIFACT_TYPE", "model"),
            token=token, allow_patterns=allow, local_dir=str(dest)))

    # Two destinations, because the artifacts split cleanly by shape and the
    # durable mount may be a BUCKET, which is good at few large objects and bad
    # at many small ones:
    #
    #   LLMDet/work_dirs   273 files, 4.94 GB, ~18 MB each  -> durable
    #   sessions/*/frames  900 files, 0.07 GB, ~78 KB each  -> ephemeral
    #
    # The frames are 77% of the files but 1.4% of the bytes. Fetching them onto
    # a mounted bucket writes 900 cache-metadata sidecars, which is what made a
    # boot stall at 25% with repeated "[Errno 5] Input/output error". They cost
    # seconds to re-fetch onto local disk, so persisting them buys nothing and
    # costs a hang. The 4.94 GB of weights is the part worth keeping across a
    # sleep, and it is exactly the shape a bucket handles well.
    weights = pull(["LLMDet/work_dirs/**"], CACHE_ROOT / "artifacts", "weights (durable)")
    session = pull(["tools/dashboard/sessions/**"], HOME / "_artifacts_session",
                   "session cache (ephemeral)")
    # The detector config addresses its text encoder and LMM by RELATIVE path
    # (grounding_dino_swin_t.py: lang_model_name = '../huggingface/bert-base-uncased/',
    # lmm = '../huggingface/my_llava-onevision-qwen2-0.5b-ov-2/'), resolved
    # against the process cwd, which is HOME. So they belong at HOME.parent,
    # NOT under the app directory. ~2.1 GB in a handful of large files, so this
    # goes to durable storage with the weights.
    models = pull(["huggingface/**"], CACHE_ROOT / "hf_models",
                  "detector models (durable)")

    for src, dest, what in ((session / "tools" / "dashboard" / "sessions", SESSIONS_DIR,
                             "session cache"),
                            (weights / "LLMDet" / "work_dirs", WORKDIRS, "work_dirs"),
                            (models / "huggingface", HF_MODELS, "detector models")):
        if not src.exists():
            print(f"[app] note: {what} not present in the artifact repo", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(src, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(src, dest)
        print(f"[app] {what} -> {dest}", flush=True)
    return True


#: Files the detector config names directly. Checked at startup because a
#: missing one otherwise surfaces ~40 s into an analysis job as an opaque
#: transformers OSError ("Incorrect path_or_model_id"), on the GPU, in a
#: background thread — about the least useful place to learn it.
DETECTOR_ASSETS = (
    "bert-base-uncased/config.json",                    # lang_model_name
    "my_llava-onevision-qwen2-0.5b-ov-2/config.json",   # lmm=
    "mediapipe/face_detection_full_range.tflite",       # head_pose_backend
)


def check_detector_assets() -> None:
    missing = [a for a in DETECTOR_ASSETS if not (HF_MODELS / a).exists()]
    if missing:
        print(f"[app] WARNING: {len(missing)} detector asset(s) missing under "
              f"{HF_MODELS} — replay works, analysing video will not:", flush=True)
        for a in missing:
            print(f"[app]   missing: {a}", flush=True)
    else:
        print(f"[app] detector assets present under {HF_MODELS}", flush=True)


def main() -> int:
    have_artifacts = fetch_artifacts()
    if have_artifacts:
        check_detector_assets()

    session = os.environ.get("SESSION", "0325")
    session_dir = SESSIONS_DIR / session
    model = os.environ.get("DASHBOARD_MODEL", "arch/mstcn_556_hp")

    # Live analysis needs the detector config and a device. Without --config the
    # Analyse button can only replay; with it, an uploaded video runs the full
    # detector -> tracker -> features -> temporal chain.
    config = os.environ.get("RUNTIME_CONFIG", "LLMDet/configs/attention_runtime.yaml")
    device = os.environ.get("DEVICE", "cuda:0")

    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[app] {device} requested but torch reports no CUDA device — "
                  f"falling back to cpu. Analysis will run at roughly 1 fps.",
                  flush=True)
            device = "cpu"
        elif device.startswith("cuda"):
            print(f"[app] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as e:
        print(f"[app] could not query torch for CUDA ({e}); using {device}", flush=True)

    argv = [sys.argv[0], "--host", "0.0.0.0", "--port", str(PORT),
            "--config", config, "--device", device]
    if have_artifacts and session_dir.exists():
        argv += ["--session", str(session_dir), "--model", model]
        print(f"[app] session {session_dir} | model {model}", flush=True)
    else:
        # Falls back to a tracked cue log so the Space always renders something
        # rather than failing to boot. Model switching is unavailable here: a
        # recorded cue log stores decisions, not features.
        replay = HOME / "tools" / "dashboard" / "demo_session.jsonl"
        argv += ["--replay", str(replay)]
        print(f"[app] replay mode: {replay}", flush=True)

    sys.path.insert(0, str(HOME / "tools" / "dashboard"))
    sys.argv = argv
    os.chdir(HOME)

    import runpy
    runpy.run_path(str(HOME / "tools" / "dashboard" / "server.py"),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
