"""Signal transforms used by Perceive (PCV, Sect. 3.1).

Every transform maps a single channel of length T to a feature series of the
same length T, so that feature index ``t`` always refers to sample ``t`` of the
original series.  Differences and the root-mean-square energy use left
replication: samples before the start of the series repeat ``x[0]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["level", "difference", "rms_energy", "apply_transform", "TRANSFORMS"]


def level(x: np.ndarray) -> np.ndarray:
    """Identity transform."""
    x = np.asarray(x, dtype=float)
    return x.copy()


def _left_replicate(x: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return x
    return np.concatenate([np.full(pad, x[0], dtype=float), x])


def difference(x: np.ndarray, direction: str = "rising") -> np.ndarray:
    """First difference with left replication.

    ``direction="rising"`` returns ``u(t) = x(t) - x(t-1)``; ``"falling"``
    returns ``u(t) = x(t-1) - x(t)``.  With left replication ``u(0) = 0``.
    """
    x = np.asarray(x, dtype=float)
    xp = _left_replicate(x, 1)
    d = xp[1:] - xp[:-1]
    if direction == "rising":
        return d
    if direction == "falling":
        return -d
    raise ValueError(f"unknown difference direction: {direction!r}")


def rms_energy(x: np.ndarray, width: int = 16) -> np.ndarray:
    """Root-mean-square energy over ``width`` samples ending at ``t``.

    The window ``[t - width + 1, t]`` is left-replicated at the boundary.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    x = np.asarray(x, dtype=float)
    xp = _left_replicate(x, width - 1)
    c = np.concatenate([[0.0], np.cumsum(xp * xp)])
    sums = c[width:] - c[:-width]
    return np.sqrt(sums / width)


TRANSFORMS = {
    "level": level,
    "rising": lambda x: difference(x, "rising"),
    "falling": lambda x: difference(x, "falling"),
    "rms16": lambda x: rms_energy(x, 16),
}


def apply_transform(name: str, x: np.ndarray) -> np.ndarray:
    try:
        fn = TRANSFORMS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown transform: {name!r}") from exc
    return fn(np.asarray(x, dtype=float))
