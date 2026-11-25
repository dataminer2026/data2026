"""Predicate margins and the generalized-mean relaxation (PCV, Sect. 3.1/3.4).

Eq. (1):   r_m(t) = (u_m(t) - mu_m) / s_m - theta_m
Eq. (7):   Maxp(z) = m * ( mean_i ([z_i]_+ / m)^p )^(1/p)   with m = max_i z_i, m > 0
           Maxp(z) = m                                       otherwise
           Minp(z) = -Maxp(-z)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Normalizer", "Concept", "gmr_max", "gmr_min", "relaxed_min", "relaxed_max"]


@dataclass(frozen=True)
class Normalizer:
    """Training median centre and training IQR scale for one concept."""

    center: float
    scale: float

    @classmethod
    def fit(cls, u: np.ndarray) -> "Normalizer":
        u = np.asarray(u, dtype=float).ravel()
        q25, med, q75 = np.percentile(u, [25.0, 50.0, 75.0])
        scale = max(float(q75 - q25), 1e-6)
        return cls(center=float(med), scale=scale)

    def margin(self, u: np.ndarray, theta: float) -> np.ndarray:
        return (np.asarray(u, dtype=float) - self.center) / self.scale - theta

    def standardize(self, u: np.ndarray) -> np.ndarray:
        return (np.asarray(u, dtype=float) - self.center) / self.scale


@dataclass(frozen=True)
class Concept:
    """A named signal feature ``m = (v, k)``.

    ``channel`` is the (zero-based) channel index, ``transform`` names an entry
    of :data:`pcv.transforms.TRANSFORMS`.  A concept is exported together with
    its transform and normalizer so that a monitor can recompute it exactly.
    """

    channel: int
    transform: str = "level"

    @property
    def name(self) -> str:
        return f"ch{self.channel}:{self.transform}"


def gmr_max(z: np.ndarray, p: float = 8.0) -> np.ndarray:
    """Generalized-mean relaxation of ``max`` that preserves the sign."""
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=-1, keepdims=True)
    positive = m > 0
    safe_m = np.where(positive, m, 1.0)
    ratio = np.maximum(z, 0.0) / safe_m
    relaxed = safe_m * np.mean(ratio**p, axis=-1, keepdims=True) ** (1.0 / p)
    return np.where(positive, relaxed, m).squeeze(-1)


def gmr_min(z: np.ndarray, p: float = 8.0) -> np.ndarray:
    """Dual of :func:`gmr_max`: ``Minp(z) = -Maxp(-z)``."""
    return -gmr_max(-np.asarray(z, dtype=float), p)


def relaxed_min(z: np.ndarray, p: float) -> np.ndarray:
    return gmr_min(z, p)


def relaxed_max(z: np.ndarray, p: float) -> np.ndarray:
    return gmr_max(z, p)
