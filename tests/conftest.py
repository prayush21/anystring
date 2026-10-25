"""Shared fixtures. Every test runs headless: no camera, mic, speakers or network.

Tests that genuinely need hardware are marked ``@pytest.mark.hardware`` and are
deselected by default (``uv run pytest -m hardware`` runs them on the MacBook).
"""

from __future__ import annotations

import numpy as np
import pytest

from anystring.config import Config


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point user files at a temp dir so no test reads or writes ~/.anystring."""
    monkeypatch.setenv("ANYSTRING_HOME", str(tmp_path / "home"))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def cfg() -> Config:
    """Default config with a fixed string orientation (slot 0 = low E)."""
    return Config().with_overrides({"strings": {"low_e_side": "neg_v"}})
