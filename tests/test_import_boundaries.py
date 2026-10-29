"""Enforces the package rule: a work package imports only ``anystring.types``,
``anystring.config`` and its own modules. This keeps WP1..WP4 buildable in parallel.

WP5 (integration: app.py, ui/, replay/, tools/latency_probe.py) is the one layer
allowed to import across packages. A new file in a shared directory (play/, capture/)
fails here until the lead assigns it an owner below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = {"anystring/types.py", "anystring/config.py"}
INTEGRATION = "WP5"

# Most specific match wins: exact file first, then the longest directory prefix.
OWNERS = {
    "anystring/__init__.py": "WP0",
    "anystring/types.py": "WP0",
    "anystring/config.py": "WP0",
    "anystring/capture/__init__.py": "WP0",
    "anystring/play/__init__.py": "WP0",
    "anystring/capture/camera.py": "WP1",
    "anystring/vision/": "WP1",
    "tests/vision/": "WP1",
    "anystring/geometry/": "WP2",
    "anystring/play/strum.py": "WP2",
    "anystring/play/chords.py": "WP2",
    "anystring/play/mute.py": "WP2",
    "anystring/play/engine.py": "WP2",
    "tests/play/": "WP2",
    "anystring/audio/": "WP3",
    "tests/audio/": "WP3",
    "anystring/capture/mic.py": "WP4",
    "anystring/play/gate.py": "WP4",
    "tests/gate/": "WP4",
    "anystring/app.py": "WP5",
    "anystring/pipeline.py": "WP5",
    "anystring/ui/": "WP5",
    "anystring/replay/": "WP5",
    "tools/latency_probe.py": "WP5",
    "tools/acceptance.py": "WP5",
    "tools/grip_sweep.py": "WP5",
    "anystring/song/__init__.py": "WP0",
    "anystring/song/": "WP7",
    "tests/song/": "WP7",
    "anystring/coach/": "WP8",
    "tests/coach/": "WP8",
}


def owner(rel: str) -> str | None:
    if rel in OWNERS:
        return OWNERS[rel]
    dirs = [k for k in OWNERS if k.endswith("/") and rel.startswith(k)]
    return OWNERS[max(dirs, key=len)] if dirs else None


def module_file(name: str, root: Path = ROOT) -> str | None:
    """Repo-relative file for a dotted module name inside the repo, else None."""
    base = root / Path(*name.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p.relative_to(root).as_posix()
    return None


def local_imports(path: Path, root: Path = ROOT) -> list[tuple[int, str]]:
    """(line, repo-relative target file) for every import of a repo module in ``path``."""
    pkg = path.relative_to(root).parts[:-1]
    found = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = pkg[: len(pkg) - (node.level - 1)]
                mod = ".".join([*anchor, node.module] if node.module else anchor)
            else:
                mod = node.module or ""
            # "from pkg import name" may name a submodule or an attribute of pkg.
            names = [
                f"{mod}.{a.name}" if module_file(f"{mod}.{a.name}", root) else mod
                for a in node.names
            ]
        else:
            continue
        for name in names:
            if f := module_file(name, root):
                found.append((node.lineno, f))
    return found


def checked_files() -> list[Path]:
    files = [*(ROOT / "anystring").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
    files += [
        p
        for d in ("vision", "play", "audio", "gate", "song", "coach")
        for p in (ROOT / "tests" / d).rglob("*.py")
    ]
    return sorted(files)


def test_every_file_has_an_owner():
    unowned = [
        p.relative_to(ROOT).as_posix()
        for p in checked_files()
        if owner(p.relative_to(ROOT).as_posix()) is None
    ]
    assert not unowned, f"assign these to a work package in {Path(__file__).name}: {unowned}"


@pytest.mark.parametrize("path", checked_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_imports_stay_inside_work_package(path: Path):
    rel = path.relative_to(ROOT).as_posix()
    me = owner(rel)
    if me in (None, INTEGRATION):
        return
    bad = []
    for line, target in local_imports(path):
        if target in SHARED or target.endswith("/__init__.py") and owner(target) == "WP0":
            continue
        if target.startswith("tests/fakes/") and rel.startswith("tests/"):
            continue
        if owner(target) != me:
            bad.append(f"line {line}: {target} ({owner(target)})")
    assert not bad, f"{rel} ({me}) may import only types.py, config.py and {me} modules: {bad}"


def test_production_code_never_imports_tests():
    bad = [
        (p.relative_to(ROOT).as_posix(), t)
        for p in (ROOT / "anystring").rglob("*.py")
        for _, t in local_imports(p)
        if t.startswith("tests/")
    ]
    assert not bad


def test_checker_catches_violations(tmp_path):
    """The rule itself works: imports of WP4's gate from play/ resolve to gate.py."""
    assert owner("anystring/play/engine.py") == "WP2"
    assert owner("anystring/play/gate.py") == "WP4"
    assert owner("anystring/vision/anything_new.py") == "WP1"
    assert owner("anystring/play/new_helper.py") is None
    play = tmp_path / "anystring" / "play"
    play.mkdir(parents=True)
    for f in (
        tmp_path / "anystring" / "__init__.py",
        tmp_path / "anystring" / "types.py",
        play / "__init__.py",
        play / "gate.py",
    ):
        f.write_text("")
    probe = play / "engine.py"
    probe.write_text(
        "from anystring.play import gate\n"
        "from . import gate as g\n"
        "from .gate import Thing\n"
        "import anystring.types\n"
        "import numpy\n"
    )
    targets = [t for _, t in local_imports(probe, tmp_path)]
    assert targets == ["anystring/play/gate.py"] * 3 + ["anystring/types.py"]
