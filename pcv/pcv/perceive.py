"""Step 1, Perceive: concepts, normalization and signed predicate margins.

For channel ``v`` and transform ``g_k`` the concept is ``m = (v, k)`` and
``u_m(t) = g_k(x_{:,v})_t``.  With the training median ``mu_m``, the training
interquartile range ``s_m`` and a threshold ``theta_m``:

    r_m(t) = (u_m(t) - mu_m) / s_m - theta_m            (Eq. 1)

Everything a monitor needs to recompute ``r_m`` (transform name, channel,
centre, scale, threshold) is exported with the artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .predicates import Concept, Normalizer
from .transforms import apply_transform

__all__ = ["Perceiver", "DEFAULT_TRANSFORMS"]


DEFAULT_TRANSFORMS = ("level", "rising", "falling", "rms16")


@dataclass
class Perceiver:
    concepts: tuple[Concept, ...]
    normalizers: dict[str, Normalizer]
    thetas: dict[str, float]
    theta_bounds: dict[str, tuple[float, float]]
    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS

    # -- construction ---------------------------------------------------
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        transforms: Sequence[str] = DEFAULT_TRANSFORMS,
        theta_quantile: float = 0.75,
    ) -> "Perceiver":
        X = _as_batch(X)
        n_channels = X.shape[2]
        concepts = tuple(Concept(v, k) for v in range(n_channels) for k in transforms)
        normalizers: dict[str, Normalizer] = {}
        thetas: dict[str, float] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for concept in concepts:
            u = np.concatenate([apply_transform(concept.transform, X[i, :, concept.channel]) for i in range(len(X))])
            norm = Normalizer.fit(u)
            z = norm.standardize(u)
            normalizers[concept.name] = norm
            thetas[concept.name] = float(np.quantile(z, theta_quantile))
            lo, hi = np.quantile(z, [0.05, 0.95])
            bounds[concept.name] = (float(lo), float(hi))
        return cls(concepts, normalizers, thetas, bounds, tuple(transforms))

    # -- use ------------------------------------------------------------
    @classmethod
    def fixed(
        cls,
        n_channels: int,
        threshold: float,
        transform: str = "level",
    ) -> "Perceiver":
        """A perceiver with identity normalization and a fixed threshold.

        Used by the controlled experiments of Sect. 4.9/4.10, where the paper
        keeps rules and thresholds fixed across methods and only the noise
        level changes: ``r_m(t) = u_m(t) - threshold``.
        """
        concepts = tuple(Concept(v, transform) for v in range(n_channels))
        return cls(
            concepts=concepts,
            normalizers={c.name: Normalizer(0.0, 1.0) for c in concepts},
            thetas={c.name: float(threshold) for c in concepts},
            theta_bounds={c.name: (0.0, 1.0) for c in concepts},
            transforms=(transform,),
        )

    def margin(self, x: np.ndarray, concept: Concept) -> np.ndarray:
        u = apply_transform(concept.transform, np.asarray(x, dtype=float)[:, concept.channel])
        return self.normalizers[concept.name].margin(u, self.thetas[concept.name])

    def margins(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("expected a single series of shape (T, V)")
        return {c.name: self.margin(x, c) for c in self.concepts}

    def project(self, concept: Concept, theta: float) -> float:
        lo, hi = self.theta_bounds[concept.name]
        return float(min(max(theta, lo), hi))

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "transforms": list(self.transforms),
            "concepts": [
                {
                    "channel": c.channel,
                    "transform": c.transform,
                    "center": self.normalizers[c.name].center,
                    "scale": self.normalizers[c.name].scale,
                    "theta": self.thetas[c.name],
                    "bounds": list(self.theta_bounds[c.name]),
                }
                for c in self.concepts
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Perceiver":
        concepts, normalizers, thetas, bounds = [], {}, {}, {}
        for row in payload["concepts"]:
            concept = Concept(int(row["channel"]), row["transform"])
            concepts.append(concept)
            normalizers[concept.name] = Normalizer(float(row["center"]), float(row["scale"]))
            thetas[concept.name] = float(row["theta"])
            bounds[concept.name] = (float(row["bounds"][0]), float(row["bounds"][1]))
        return cls(tuple(concepts), normalizers, thetas, bounds, tuple(payload["transforms"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Perceiver":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _as_batch(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return X[None, ...]
    if X.ndim != 3:
        raise ValueError("expected shape (T, V) or (N, T, V)")
    return X
