"""Step 2, Compile: structure search over shared-origin rule graphs (PCV, Sect. 3.4).

The trainer here is a simplified variant, not the pipeline that produced the
UEA tables in Sect. 5.  It keeps the two ingredients that matter for the method:

* thresholds are optimised on the **relaxed** score and projected onto the
  training 5th-95th percentile range (:meth:`Perceiver.project`);
* structures are selected on the **hard** score, with a bounded beam that
  deduplicates graphs and prefers shorter rules on ties.

Candidate atoms are canonical: the left window starts at the origin, so a
relation is fixed by ``(concept_left, width_left, concept_right, width_right,
gap)``.  Additional atoms may join an existing window, so a ``Before(B,C)`` can
reuse the ``B`` of a ``Before(A,B)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .graph import And, Atom, Root, RuleSet, windows_of
from .perceive import Perceiver
from .predicates import Concept

__all__ = ["FitConfig", "fit", "balanced_accuracy", "MarginBank"]


@dataclass
class FitConfig:
    transforms: Sequence[str] = ("level", "rising", "falling", "rms16")
    widths: Sequence[int] = (4, 8, 16)
    gaps: Sequence[int] = (4, 12, 24)
    max_atoms: int = 3
    beam: int = 8
    pool: int = 48
    kappa: float = 10.0
    lo: int = 0
    hi: int = 24
    min_true: int = 2
    refine_thresholds: bool = False
    theta_grid: Sequence[float] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)


class MarginBank:
    """Vectorised window scores over a batch of series, shape ``(N, T)``."""

    def __init__(self, perceiver: Perceiver, X: np.ndarray) -> None:
        self.perceiver = perceiver
        self.margins = np.stack([np.stack([perceiver.margins(x)[c.name] for c in perceiver.concepts], axis=-1) for x in X])
        self.index = {c.name: i for i, c in enumerate(perceiver.concepts)}
        self.length = X.shape[1]
        self._cache: dict[tuple, np.ndarray] = {}

    def window(self, concept: Concept, a: int, b: int) -> np.ndarray:
        key = (concept.name, a, b)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        M = self.margins[:, :, self.index[concept.name]]
        width = b - a + 1
        out = np.full(M.shape, -np.inf, dtype=np.float32)
        if width <= self.length:
            windows = sliding_window_view(M, width, axis=1).min(axis=-1)
            n = windows.shape[1] - a
            if n > 0:
                out[:, :n] = windows[:, a : a + n]
        self._cache[key] = out
        return out

    def atom(self, atom: Atom) -> np.ndarray:
        left = self.window(atom.left.concept, atom.left.a, atom.left.b)
        right = self.window(atom.right.concept, atom.right.a, atom.right.b)
        bound = atom.cap if atom.kind == "before" else None
        if bound is None:  # meets / overlaps are origin-dependent; evaluate per origin
            values = np.empty_like(left)
            for t in range(self.length):
                margin = atom.endpoint_margin(t)
                values[:, t] = np.minimum(np.minimum(left[:, t], right[:, t]), margin)
            return values
        return np.minimum(np.minimum(left, right), np.float32(bound))

    def body(self, node) -> np.ndarray:
        if isinstance(node, Atom):
            return self.atom(node)
        values = [self.body(c) for c in node.children]
        stacked = np.stack(values)
        return stacked.min(axis=0) if isinstance(node, And) else stacked.max(axis=0)

    def root(self, root: Root) -> np.ndarray:
        values = self.body(root.body)
        return values.max(axis=1)


def balanced_accuracy(scores: np.ndarray, labels: np.ndarray, positive: int) -> float:
    pred = scores > 0
    truth = labels == positive
    if truth.sum() == 0 or (~truth).sum() == 0:
        return 0.0
    tpr = float(np.mean(pred[truth]))
    tnr = float(np.mean(~pred[~truth]))
    return 0.5 * (tpr + tnr)


def _canonical(size_a: int, size_b: int, gap: int) -> tuple[int, int, int, int]:
    return 0, size_a - 1, size_a + gap, size_a + gap + size_b - 1


def _propose(bank: MarginBank, concepts: Sequence[Concept], cfg: FitConfig) -> list[Atom]:
    atoms: list[Atom] = []
    for c1, c2 in product(concepts, repeat=2):
        if c1 == c2:
            continue
        for w1, w2, gap in product(cfg.widths, cfg.widths, cfg.gaps):
            a1, b1, a2, b2 = _canonical(w1, w2, gap)
            if b2 >= bank.length:
                continue
            atoms.append(
                Atom("before", _window(c1, a1, b1), _window(c2, a2, b2), cfg.lo, cfg.hi, cfg.kappa)
            )
    return atoms


def _window(concept: Concept, a: int, b: int):
    from .windows import WindowRef

    return WindowRef(concept, a, b)


def _chain(body, bank: MarginBank, concepts: Sequence[Concept], cfg: FitConfig) -> list[Atom]:
    """Atoms whose left window is an existing window of ``body``."""
    atoms: list[Atom] = []
    for window in windows_of(body):
        for c2 in concepts:
            if c2 == window.concept:
                continue
            for w2, gap in product(cfg.widths, cfg.gaps):
                a2 = window.b + gap + 1
                b2 = a2 + w2 - 1
                if b2 >= bank.length:
                    continue
                atoms.append(Atom("before", window, _window(c2, a2, b2), cfg.lo, cfg.hi, cfg.kappa))
    return atoms


def _as_and(children: Iterable) -> And:
    return And(tuple(children))


def fit(
    X: np.ndarray,
    y: Sequence,
    config: FitConfig | None = None,
    max_series: int | None = 400,
    seed: int = 11,
) -> tuple[RuleSet, Perceiver, dict]:
    """Learn one root per class from labelled multivariate series ``(N, T, V)``."""
    cfg = config or FitConfig()
    X = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    classes = list(dict.fromkeys(labels.tolist()))
    rng = np.random.default_rng(seed)
    if max_series is not None and len(X) > max_series:
        keep = rng.choice(len(X), size=max_series, replace=False)
        X, labels = X[keep], labels[keep]

    perceiver = Perceiver.fit(X, transforms=cfg.transforms)
    bank = MarginBank(perceiver, X)
    pool = _propose(bank, perceiver.concepts, cfg)

    roots: list[Root | None] = []
    diagnostics: dict = {"atoms": []}

    for c in classes:
        ranked = sorted(
            ((balanced_accuracy(bank.root(Root(a)), labels, c), a) for a in pool),
            key=lambda row: -row[0],
        )
        ranked = [row for row in ranked if row[0] > 0.5][: cfg.pool]
        if not ranked:
            # No candidate separates this class from the rest; a root that never
            # fires is safer than an arbitrary one.  This happens for classes
            # defined by the *absence* of a pattern, which one-vs-rest
            # conjunctions cannot express.
            roots.append(None)
            diagnostics["atoms"].append([])
            continue

        beam: list[tuple[float, And]] = [(ranked[0][0], _as_and([ranked[0][1]]))]
        best_score, best_body = beam[0]
        for _ in range(cfg.max_atoms - 1):
            candidates: list[tuple[float, And]] = []
            for score, body in beam:
                extra = _chain(body, bank, perceiver.concepts, cfg)
                for atom in extra[:1000]:
                    if any(a == atom for a in body.children):
                        continue
                    grown = _as_and(list(body.children) + [atom])
                    value = balanced_accuracy(bank.root(Root(grown)), labels, c)
                    candidates.append((value, grown))
            if not candidates:
                break
            candidates.sort(key=lambda row: (-row[0], len(row[1].children)))
            beam = candidates[: cfg.beam]
            if beam[0][0] > best_score:
                best_score, best_body = beam[0]
        roots.append(Root(best_body))
        diagnostics["atoms"].append([str(a) for a in best_body.children])

    if cfg.refine_thresholds:
        _refine(perceiver, bank, X, labels, classes, roots, cfg)

    return RuleSet(tuple(roots)), perceiver, diagnostics


def _refine(perceiver, bank, X, labels, classes, roots, cfg: FitConfig) -> None:
    """Coordinate search on ``theta`` with projection to the training bounds.

    Only concepts that occur in a selected root are visited; changing the
    threshold of an unused concept cannot change the decision.
    """
    used = {window.concept.name for root in roots if root is not None for window in windows_of(root)}
    for concept in perceiver.concepts:
        if concept.name not in used:
            continue
        column = bank.margins[:, :, bank.index[concept.name]]
        standardized = perceiver.normalizers[concept.name].standardize(column)
        original = perceiver.thetas[concept.name]
        best = (_objective(perceiver, X, labels, classes, roots), original)
        for quantile in cfg.theta_grid:
            theta = perceiver.project(concept, float(np.quantile(standardized, quantile)))
            perceiver.thetas[concept.name] = theta
            score = _objective(perceiver, X, labels, classes, roots)
            if score > best[0] + 1e-9:
                best = (score, theta)
        perceiver.thetas[concept.name] = best[1]


def _objective(perceiver, X, labels, classes, roots) -> float:
    ruleset = RuleSet(tuple(roots))
    correct = 0
    for x, label in zip(X, labels):
        predicted, _ = ruleset.predict(perceiver.margins(x))
        predicted_label = classes[predicted] if predicted >= 0 else None
        correct += int(predicted_label == label)
    return correct / len(X)
