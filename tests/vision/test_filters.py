from __future__ import annotations

import math

import numpy as np
import pytest

from racketguitar.config import FilterConfig
from racketguitar.vision.filters import OneEuroFilter, fingertip_filter


def _times(n: int, fps: float = 60.0, jitter: float = 0.0, rng=None) -> np.ndarray:
    dt = np.full(n, 1.0 / fps)
    if jitter:
        dt = dt * (1.0 + rng.uniform(-jitter, jitter, n))
    return 100.0 + np.cumsum(dt)  # monotonic-style clock, not starting at 0


def test_first_sample_passes_through():
    f = OneEuroFilter(1.0, 0.1)
    x = np.array([[3.0, 4.0], [5.0, 6.0]])
    out = f(x, 10.0)
    np.testing.assert_array_equal(out, x)
    assert out is not x
    assert f.initialised


def test_constant_signal_is_unchanged(rng):
    f = OneEuroFilter(1.5, 0.05)
    x = rng.uniform(0, 1280, (21, 2))
    for t in _times(200, jitter=0.4, rng=rng):
        out = f(x, t)
    np.testing.assert_allclose(out, x, atol=1e-9)


def test_dt_zero_and_backwards_are_safe():
    f = OneEuroFilter(1.0, 0.1)
    f(np.array([0.0]), 1.0)
    a = f(np.array([10.0]), 1.1)
    for t in (1.1, 1.05):  # repeated, then out-of-order timestamp
        b = f(np.array([1e6]), t)
        assert np.all(np.isfinite(b))
        np.testing.assert_array_equal(b, a)
    c = f(np.array([10.0]), 1.2)  # state continues from the last good sample
    assert 0.0 < c[0] <= 10.0


def test_reduces_jitter_on_static_noise(rng):
    f = OneEuroFilter(1.0, 0.05)
    truth = np.array([[640.0, 360.0]] * 4)
    ts = _times(600, jitter=0.3, rng=rng)
    raw = truth + rng.normal(0, 1.0, (len(ts), 4, 2))
    out = np.array([f(x, t) for x, t in zip(raw, ts, strict=True)])
    settled = slice(120, None)
    raw_std = np.std(raw[settled] - truth)
    out_std = np.std(out[settled] - truth)
    # beta 0.05 trades a little jitter for much less lag on moving input: ~0.31x here.
    assert out_std < 0.4 * raw_std


def test_lag_on_fast_motion_falls_as_beta_rises():
    ts = _times(120)
    speed = 2000.0  # px/s, a fast strum
    xs = speed * (ts - ts[0])
    lags = []
    for beta in (0.0, 0.01, 0.1, 1.0):
        f = OneEuroFilter(1.5, beta)
        out = [f(np.array([x]), t)[0] for x, t in zip(xs, ts, strict=True)]
        lags.append(xs[-1] - out[-1])
    assert all(a > b for a, b in zip(lags, lags[1:], strict=False)), lags
    assert lags[-1] < 0.05 * lags[0]
    # With a large beta the steady-state lag is a few pixels at 2000 px/s.
    assert lags[-1] < 5.0


def test_variable_dt_matches_closed_form_with_beta_zero():
    # With beta = 0 it is a plain first-order low-pass whose alpha depends on dt.
    f = OneEuroFilter(2.0, 0.0)
    f(np.array([0.0]), 0.0)
    y, t = 0.0, 0.0
    for dt in (0.01, 0.05, 0.002, 0.03):
        t += dt
        tau = 1 / (2 * math.pi * 2.0)
        a = 1 / (1 + tau / dt)
        y = y + a * (1.0 - y)
        assert f(np.array([1.0]), t)[0] == pytest.approx(y)


def test_elements_are_independent():
    f = OneEuroFilter(1.0, 0.0)
    f(np.zeros((2, 2)), 0.0)
    out = f(np.array([[0.0, 0.0], [10.0, 0.0]]), 1 / 60)
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0
    assert 0.0 < out[1, 0] < 10.0


def test_reset_and_shape_change():
    f = OneEuroFilter(1.0, 0.1)
    f(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        f(np.zeros(4), 0.1)
    f.reset()
    assert not f.initialised
    np.testing.assert_array_equal(f(np.ones(4), 0.2), np.ones(4))


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        OneEuroFilter(0.0, 0.1)
    with pytest.raises(ValueError):
        OneEuroFilter(1.0, -0.1)


def test_config_factories():
    cfg = FilterConfig()
    f = fingertip_filter(cfg)
    assert (f.min_cutoff, f.beta, f.d_cutoff) == (
        cfg.fingertip_min_cutoff,
        cfg.fingertip_beta,
        cfg.d_cutoff,
    )
