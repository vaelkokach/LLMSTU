"""The vendored trees must be complete in a fresh clone.

These do not test behaviour. They test that files which exist on a working
machine are actually *tracked*, because the repo has now been bitten twice by
the same thing: `.gitignore` carried an unanchored `datasets/`, git patterns
without a leading slash match at any depth, and two source directories were
silently excluded from every clone:

    LLMDet/mmdet/datasets/                    (53 files)
    LLMDet/configs/_base_/datasets/           (coco_detection.py)

Neither was noticeable locally — on a machine that had once run the pipeline the
directories are simply there, untracked — so both surfaced only when a Hugging
Face Space built from the repo. The first died at `import mmdet.datasets`, the
second at a FileNotFoundError for a `_base_` config, inside a background
analysis job, minutes into a GPU run.

Cheap, offline, no imports of the heavy stack.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "LLMDet" / "configs"
MMDET = REPO / "LLMDet" / "mmdet"


def _base_targets(cfg: Path):
    """The paths a config's `_base_ = ...` refers to, resolved."""
    try:
        tree = ast.parse(cfg.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "_base_" for t in node.targets)):
            continue
        value = node.value
        if isinstance(value, ast.Constant):
            items = [value]
        elif isinstance(value, (ast.List, ast.Tuple)):
            items = [e for e in value.elts if isinstance(e, ast.Constant)]
        else:
            continue
        out += [(cfg.parent / e.value).resolve()
                for e in items if isinstance(e.value, str)]
    return out


@pytest.mark.skipif(not CONFIGS.is_dir(), reason="configs tree absent")
def test_every_config_base_resolves():
    """No config may inherit from a file that is not in the repo."""
    missing = []
    for cfg in sorted(CONFIGS.rglob("*.py")):
        for target in _base_targets(cfg):
            if not target.exists():
                missing.append(f"{cfg.relative_to(REPO).as_posix()} -> {target}")
    assert not missing, (
        "config(s) inherit from files missing in this clone:\n  "
        + "\n  ".join(missing)
        + "\n\nIf they exist on your machine but not in `git ls-files`, check "
          ".gitignore for an unanchored directory pattern."
    )


@pytest.mark.skipif(not MMDET.is_dir(), reason="vendored mmdet absent")
@pytest.mark.parametrize("sub", ["datasets", "models", "apis", "structures",
                                 "evaluation", "engine", "utils",
                                 "visualization"])
def test_mmdet_subpackage_present(sub):
    """register_all_modules() imports these; a missing one is ModuleNotFoundError."""
    pkg = MMDET / sub / "__init__.py"
    assert pkg.is_file(), (
        f"LLMDet/mmdet/{sub}/ is missing from this clone. It is imported by "
        f"mmdet.utils.setup_env.register_all_modules(), which the detector "
        f"calls on first use."
    )


@pytest.mark.skipif(not (REPO / ".gitignore").is_file(), reason="no .gitignore")
def test_no_unanchored_pattern_hides_vendored_source():
    """A bare `foo/` rule matches at any depth — keep it away from source trees.

    Checked by name rather than by running git, so it holds in a checkout with
    no git available. Only directory names that actually occur inside the
    vendored trees are policed; unanchored rules for build noise are fine.
    """
    # Directory names that appear inside source trees but are never source.
    NEVER_SOURCE = {"__pycache__", ".pytest_cache", ".mypy_cache",
                    ".ipynb_checkpoints", ".venv", "node_modules", ".git"}

    vendored_dirnames = set()
    for tree in (MMDET, CONFIGS, REPO / "LLMDet" / "llava", REPO / "LLMDet" / "ram"):
        if tree.is_dir():
            vendored_dirnames |= {d.name for d in tree.rglob("*")
                                  if d.is_dir() and d.name not in NEVER_SOURCE}

    offenders = []
    for i, raw in enumerate((REPO / ".gitignore").read_text(
            encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.startswith("/") or "/" in line.rstrip("/"):
            continue                      # anchored, or already a path
        name = line.rstrip("/")
        if name in vendored_dirnames:
            offenders.append(f".gitignore:{i}: {line!r} also matches "
                             f"a directory inside a vendored tree")

    assert not offenders, "\n".join(offenders) + (
        "\n\nAnchor it (`/name/`) or spell the data path out in full."
    )
