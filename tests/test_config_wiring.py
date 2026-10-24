"""Every Config and Calibration key is read by some production code, so no knob is dead.

Keys whose reader does not exist yet go in PENDING.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from racketguitar.config import Calibration, Config

ROOT = Path(__file__).resolve().parents[1]

PENDING: set[str] = set()  # keys waiting for code that does not exist yet


def _attributes_read() -> set[str]:
    names: set[str] = set()
    for path in [*(ROOT / "racketguitar").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]:
        if path.name == "config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
    return names


def test_every_config_key_is_read():
    read = _attributes_read()
    cfg = Config()
    keys = {f"{s.name}.{k.name}" for s in fields(cfg) for k in fields(getattr(cfg, s.name))}
    keys |= {f"calibration.{k.name}" for k in fields(Calibration)}
    unread = sorted(k for k in keys - PENDING if k.split(".")[1] not in read)
    assert not unread, f"config keys no code reads: {unread}"


def test_pending_keys_still_exist():
    cfg = Config()
    for key in PENDING:
        section, name = key.split(".")
        assert hasattr(getattr(cfg, section), name), key
