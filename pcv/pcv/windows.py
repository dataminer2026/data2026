"""Windows and window scores (PCV, Eq. 2).

A window is ``e = Win(m, [a, b])`` with offsets relative to a shared origin:

    q_e(t) = min_{j = a..b} r_m(t + j)

A repeated window is identified by its concept and offsets, so two relations
bind to the same occurrence of ``B``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .predicates import Concept

__all__ = ["WindowRef", "window_score", "window_scores_all", "sliding_min"]


@dataclass(frozen=True)
class WindowRef:
    """``Win(m, [a, b])``: a concept plus inclusive offsets from an origin."""

    concept: Concept
    a: int
    b: int

    def __post_init__(self) -> None:
        if self.a < 0:
            raise ValueError("window offsets must be non-negative")
        if self.b < self.a:
            raise ValueError("window requires a <= b")

    @property
    def width(self) -> int:
        return self.b - self.a + 1

    def start(self, origin: int) -> int:
        return origin + self.a

    def stop(self, origin: int) -> int:
        return origin + self.b

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.concept.name}[{self.a},{self.b}]"


def window_score(margin: np.ndarray, window: WindowRef, origin: int) -> float:
    """``q_e(t)`` for a single origin, with a bounds check."""
    start = window.start(origin)
    stop = window.stop(origin)
    if start < 0 or stop >= len(margin):
        raise IndexError("window runs outside the series")
    return float(np.min(margin[start : stop + 1]))


def sliding_min(values: np.ndarray, width: int) -> np.ndarray:
    """Minimum over every window of ``width`` samples, in O(n).

    Element ``i`` of the result is ``min(values[i - width + 1 : i + 1])``;
    entries whose window would start before index 0 are ``nan``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if width < 1 or n == 0:
        return out
    dq: deque[int] = deque()
    for i in range(n):
        while dq and dq[0] <= i - width:
            dq.popleft()
        while dq and values[dq[-1]] >= values[i]:
            dq.pop()
        dq.append(i)
        if i >= width - 1:
            out[i] = values[dq[0]]
    return out


def window_scores_all(margin: np.ndarray, window: WindowRef) -> np.ndarray:
    """``q_e(t)`` for every legal origin; ``nan`` where the window overflows."""
    margin = np.asarray(margin, dtype=float)
    n = len(margin)
    width = window.width
    out = np.full(n, np.nan)
    mins = sliding_min(margin, width)  # mins[i] = min(margin[i-width+1 .. i])
    last_origin = n - 1 - window.b
    if last_origin < 0:
        return out
    idx = np.arange(last_origin + 1) + window.b
    out[: last_origin + 1] = mins[idx]
    return out
