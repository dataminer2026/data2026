"""Synthetic benchmarks used by the controlled experiments (PCV, Sect. 4.9/4.10).

The device-cycle task is fully specified in the paper: three channels, 64
samples, 4-sample unit pulses, windows ``[0,3] [8,11] [16,19]`` relative to a
common origin, a four-sample gap and a cap of 0.35.  Positive series contain a
complete A-B-C cycle; the three negative families contain only A,B, only B,C, or
two separated episodes that satisfy the two relations at different occurrences
of B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DeviceCycle",
    "device_cycle",
    "device_rule_graphs",
    "gap_series",
    "load_ts",
    "DEVICE_WINDOWS",
    "DEVICE_GAP",
    "DEVICE_CAP",
]

DEVICE_WINDOWS = ((0, 3), (8, 11), (16, 19))
DEVICE_GAP = 4
DEVICE_CAP = 0.35
DEVICE_LENGTH = 64
PULSE = 4


@dataclass
class DeviceCycle:
    X: np.ndarray  # (N, T, 3)
    y: np.ndarray  # 1 = complete cycle, 0 = incomplete
    family: np.ndarray  # "positive" | "ab" | "bc" | "split"
    origin: np.ndarray


def _paint(x: np.ndarray, channel: int, start: int, length: int = PULSE, value: float = 1.0) -> None:
    if start < 0:
        raise ValueError("negative start")
    stop = min(start + length, len(x))
    x[start:stop, channel] = np.maximum(x[start:stop, channel], value)


def device_cycle(
    n_per_family: int = 200,
    sigma: float = 0.0,
    seed: int = 11,
    length: int = DEVICE_LENGTH,
) -> DeviceCycle:
    """Generate the three-channel composition benchmark of Sect. 4.9."""
    rng = np.random.default_rng(seed)
    families = ("positive", "ab", "bc", "split")
    X, y, fam, origins = [], [], [], []

    for name in families:
        for _ in range(n_per_family):
            x = np.zeros((length, 3), dtype=float)
            if name == "positive":
                t = int(rng.integers(0, length - DEVICE_WINDOWS[2][1]))
                _paint(x, 0, t)
                _paint(x, 1, t + 8)
                _paint(x, 2, t + 16)
                origin = t
            elif name == "ab":
                t = int(rng.integers(0, length - DEVICE_WINDOWS[1][1]))
                _paint(x, 0, t)
                _paint(x, 1, t + 8)
                origin = t
            elif name == "bc":
                t = int(rng.integers(0, length - DEVICE_WINDOWS[2][1]))
                _paint(x, 1, t + 8)
                _paint(x, 2, t + 16)
                origin = t
            else:  # two episodes that both satisfy a relation at a different B
                t = int(rng.integers(0, length - 32))
                _paint(x, 0, t)
                _paint(x, 1, t + 8)
                _paint(x, 1, t + 20)
                _paint(x, 2, t + 28)
                origin = t
            X.append(x)
            y.append(1 if name == "positive" else 0)
            fam.append(name)
            origins.append(origin)

    X = np.stack(X)
    if sigma > 0:
        X = X + rng.normal(0.0, sigma, size=X.shape)
    return DeviceCycle(X=X, y=np.array(y), family=np.array(fam), origin=np.array(origins))


def device_rule_graphs():
    """The two single relations and the shared-origin conjunction of Sect. 4.9.

    ``R1`` checks A before B, ``R2`` checks B before C, and ``shared`` puts both
    atoms at one origin.  The "separate origins" control is not a graph: it
    takes the minimum of the two independently maximised root scores.
    """
    from .graph import And, Atom, Root, WindowRef
    from .predicates import Concept

    ch = [Concept(i, "level") for i in range(3)]
    windows = [WindowRef(ch[i], a, b) for i, (a, b) in enumerate(DEVICE_WINDOWS)]
    r1 = Atom("before", windows[0], windows[1], lo=0, hi=24, kappa=10.0)
    r2 = Atom("before", windows[1], windows[2], lo=0, hi=24, kappa=10.0)
    return {"R1": Root(r1), "R2": Root(r2), "shared": Root(And((r1, r2)))}


def gap_series(
    n: int = 120,
    gap: int = 12,
    sigma: float = 0.2,
    seed: int = 11,
    length: int = 128,
    plateau: int = 8,
    background: float = -0.55,
    event: float = 1.0,
):
    """Two-channel series with two plateaus separated by ``gap`` samples."""
    rng = np.random.default_rng(seed)
    X = np.full((n, length, 2), background, dtype=float)
    starts = rng.integers(0, length - (2 * plateau + gap), size=n)
    spans = []
    for i, s in enumerate(starts):
        first = (int(s), int(s) + plateau - 1)
        second = (int(s) + plateau + gap, int(s) + 2 * plateau + gap - 1)
        X[i, first[0] : first[1] + 1, :] = event
        X[i, second[0] : second[1] + 1, :] = event
        spans.append((first, second))
    if sigma > 0:
        X = X + rng.normal(0.0, sigma, size=X.shape)
    return X, np.array(spans)


def load_ts(path) -> tuple[np.ndarray, np.ndarray | None]:
    """Minimal loader for the UEA ``.ts`` format (single split file).

    Each data row is ``label:ch1:ch2:...`` with comma-separated samples.
    Returns ``(X, y)`` with ``X`` of shape ``(N, T, V)``; labels are ``None``
    when the file uses the ``?`` placeholder.  Series are assumed equal length.
    """
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("@problemName"):
        raise ValueError("not a UEA .ts file")
    start = next(i for i, line in enumerate(lines) if line.startswith("@data"))

    series, labels = [], []
    for line in lines[start + 1 :]:
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        parts = line.split(":")
        labels.append(parts[0].strip())
        channels = [[float(v) for v in part.split(",") if v != ""] for part in parts[1:]]
        series.append(np.array(channels, dtype=float).T)
    if not series:
        raise ValueError("no instances found after @data")
    X = np.stack(series)
    y = None if any(label == "?" for label in labels) else np.array(labels)
    return X, y
