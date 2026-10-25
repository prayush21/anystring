"""One Euro filter (Casiez, Roussel and Vogel, CHI 2012), vectorised over arrays.

The filter is an adaptive low-pass: at low speed the cutoff sits near ``min_cutoff``
and removes jitter, and as speed rises the cutoff grows by ``beta * |speed|`` so fast
motion passes with little lag. It is driven by real timestamps, so a variable frame
interval is handled correctly. Every element of the input array is filtered
independently, e.g. a (21, 2) array of hand landmarks in pixels.
"""

from __future__ import annotations

import math

import numpy as np

from anystring.config import FilterConfig

__all__ = ["OneEuroFilter", "fingertip_filter"]


def _alpha(dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
    """Smoothing factor of a first-order low-pass with ``cutoff`` Hz over ``dt`` s."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filter an array-valued signal sample by sample.

    ``min_cutoff`` and ``d_cutoff`` are in Hz; ``beta`` is in 1/(signal units), i.e. per
    pixel for pixel coordinates. The first sample (and the first after ``reset``) passes
    through unchanged. A sample with ``dt <= 0`` (a repeated or out-of-order timestamp)
    returns the previous output and leaves the state untouched.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None  # last filtered value
        self._dx: np.ndarray | None = None  # last filtered derivative
        self._t: float | None = None

    @property
    def initialised(self) -> bool:
        return self._x is not None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t`` (seconds); returns a new array."""
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), float(t)
            return x.copy()
        if x.shape != self._x.shape:
            raise ValueError(f"shape changed from {self._x.shape} to {x.shape}; call reset()")
        dt = float(t) - self._t
        if dt <= 0.0:
            return self._x.copy()
        dx = (x - self._x) / dt
        dx_hat = self._dx + _alpha(dt, self.d_cutoff) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        x_hat = self._x + _alpha(dt, cutoff) * (x - self._x)
        self._x, self._dx, self._t = x_hat, dx_hat, float(t)
        return x_hat.copy()


def fingertip_filter(cfg: FilterConfig) -> OneEuroFilter:
    """Filter for hand landmarks, using the ``fingertip_*`` settings."""
    return OneEuroFilter(cfg.fingertip_min_cutoff, cfg.fingertip_beta, cfg.d_cutoff)
