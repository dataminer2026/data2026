"""Typed rule graph, shared-origin evaluation and classification (PCV, Sect. 3.3).

Nodes are interned by operator, child ids, offsets, relation parameters and
concept id, so a window used by two relations is one node.  Binary connectives
order their children canonically and preserve parenthesisation, because the
relaxed operators can be non-associative.

Conjunction and disjunction combine atom scores by pointwise minimum and
maximum at a common origin; a root is ``F_I psi`` with ``h_c = max_t v_psi(t)``.
Evaluating at one origin forbids satisfying the relations of a rule at
different occurrences of the window they share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .relations import before_cap, eq_margin, le_margin, lt_margin
from .windows import WindowRef, window_scores_all

__all__ = [
    "Atom",
    "And",
    "Or",
    "Root",
    "RuleSet",
    "Evaluation",
    "node_key",
    "legal_origins",
    "windows_of",
]


def node_key(node) -> tuple:
    """Stable structural key used for interning and canonical ordering."""
    if isinstance(node, Atom):
        return (
            "atom",
            node.kind,
            node.left.concept.name,
            node.left.a,
            node.left.b,
            node.right.concept.name,
            node.right.a,
            node.right.b,
            node.lo,
            node.hi,
            round(node.kappa, 12),
        )
    if isinstance(node, (And, Or)):
        tag = "and" if isinstance(node, And) else "or"
        return (tag,) + tuple(node_key(c) for c in node.children)
    raise TypeError(f"not a body node: {type(node)!r}")


@dataclass(frozen=True)
class Atom:
    """``Rel_{R,Delta}(e_i, e_j)``: two windows bound at one origin.

    Eq. (4) is ``min{ q_ei(t), q_ej(t), bR }``; the cap ``bR`` is fixed by the
    offsets and is therefore a constant of the atom.
    """

    kind: str
    left: WindowRef
    right: WindowRef
    lo: int = 0
    hi: int = 24
    kappa: float = 10.0

    def __post_init__(self) -> None:
        if self.kind not in {"before", "meets", "overlaps"}:
            raise ValueError(f"unknown relation: {self.kind!r}")
        if self.kind == "before" and self.right.a <= self.left.b:
            raise ValueError("Before requires the right window to start after the left ends")

    @property
    def gap(self) -> int | None:
        if self.kind != "before":
            return None
        return self.right.a - self.left.b - 1

    @property
    def cap(self) -> float | None:
        if self.kind != "before":
            return None
        return before_cap(self.gap, self.lo, self.hi, self.kappa)

    @property
    def max_offset(self) -> int:
        return max(self.left.b, self.right.b)

    def endpoint_margin(self, origin: int) -> float:
        ls, le = self.left.start(origin), self.left.stop(origin)
        rs, re = self.right.start(origin), self.right.stop(origin)
        if self.kind == "before":
            return lt_margin(le, rs)
        if self.kind == "meets":
            return eq_margin(le + 1, rs)
        return min(lt_margin(ls, rs), le_margin(rs, le), lt_margin(le, re))

    def value(self, left_q: float, right_q: float, origin: int) -> float:
        bound = self.cap if self.kind == "before" else None
        if bound is None:
            return float(min(left_q, right_q, self.endpoint_margin(origin)))
        return float(min(left_q, right_q, bound))


def _canonical(children: Iterable) -> tuple:
    unique: dict[tuple, object] = {}
    for child in children:
        unique.setdefault(node_key(child), child)
    return tuple(sorted(unique.values(), key=node_key))


@dataclass(frozen=True)
class And:
    children: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", _canonical(self.children))


@dataclass(frozen=True)
class Or:
    children: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", _canonical(self.children))


@dataclass(frozen=True)
class Root:
    """``phi_c = F_{I_c} psi_c``."""

    body: object
    lo: int = 0
    hi: int = 10**6

    @property
    def max_offset(self) -> int:
        return _max_offset(self.body)


def _max_offset(node) -> int:
    if isinstance(node, Atom):
        return node.max_offset
    return max(_max_offset(c) for c in node.children)


def windows_of(node) -> list[WindowRef]:
    """Distinct windows of a body or root, in canonical order."""
    seen: dict[WindowRef, None] = {}
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Root):
            stack.append(cur.body)
        elif isinstance(cur, Atom):
            seen.setdefault(cur.left, None)
            seen.setdefault(cur.right, None)
        else:
            stack.extend(cur.children)
    return sorted(seen, key=lambda w: (w.concept.name, w.a, w.b))


def legal_origins(root: Root, length: int) -> np.ndarray:
    """Origins at which every required window fits inside the series."""
    last = length - 1 - root.max_offset
    if last < 0:
        return np.empty(0, dtype=int)
    hi = min(root.hi, last)
    if hi < root.lo:
        return np.empty(0, dtype=int)
    return np.arange(root.lo, hi + 1)


def _series_length(margins: dict[str, np.ndarray]) -> int:
    return len(next(iter(margins.values())))


@dataclass
class Evaluation:
    """Hard scores of one series, plus cache statistics."""

    scores: np.ndarray
    origins: list[int | None]
    legality: list[int]
    window_values: int = 0
    body_values: int = 0

    @property
    def best(self) -> int:
        return int(np.argmax(self.scores))


class _Evaluator:
    def __init__(self, margins: dict[str, np.ndarray], precompute: bool) -> None:
        self.margins = margins
        self.precompute = precompute
        self._full: dict[WindowRef, np.ndarray] = {}
        self._window: dict[tuple[WindowRef, int], float] = {}
        self._body: dict[tuple[tuple, int], float] = {}
        self.window_values = 0
        self.body_values = 0

    def window_value(self, window: WindowRef, origin: int) -> float:
        key = (window, origin)
        cached = self._window.get(key)
        if cached is not None:
            return cached
        if window in self._full:
            table = self._full[window]
            if origin < 0 or origin >= len(table) or np.isnan(table[origin]):
                raise IndexError("window runs outside the series")
            value = float(table[origin])
        else:
            series = self.margins[window.concept.name]
            start, stop = window.start(origin), window.stop(origin)
            if start < 0 or stop >= len(series):
                raise IndexError("window runs outside the series")
            value = float(np.min(series[start : stop + 1]))
            self.window_values += 1
        self._window[key] = value
        return value

    def body_value(self, node, origin: int) -> float:
        key = (node_key(node), origin)
        cached = self._body.get(key)
        if cached is not None:
            return cached
        if isinstance(node, Atom):
            value = node.value(
                self.window_value(node.left, origin),
                self.window_value(node.right, origin),
                origin,
            )
        elif isinstance(node, And):
            value = min(self.body_value(c, origin) for c in node.children)
        else:
            value = max(self.body_value(c, origin) for c in node.children)
        self._body[key] = value
        self.body_values += 1
        return value

    def prepare(self, roots: Sequence[Root]) -> None:
        if not self.precompute:
            return
        for root in roots:
            for window in windows_of(root):
                if window not in self._full:
                    table = window_scores_all(self.margins[window.concept.name], window)
                    self._full[window] = table
                    self.window_values += int(np.sum(~np.isnan(table)))

    def root_scores(self, root: Root, origins: Sequence[int]) -> np.ndarray:
        return np.array([self.body_value(root.body, int(t)) for t in origins], dtype=float)


@dataclass(frozen=True)
class RuleSet:
    """Ordered roots, one per class.  Ties resolve to the stored order.

    A ``None`` root marks a class without positive evidence in the candidate
    pool; it never fires and never wins the argmax.
    """

    roots: tuple[Root | None, ...]

    @property
    def n_classes(self) -> int:
        return len(self.roots)

    def evaluate(self, margins: dict[str, np.ndarray], mode: str = "union") -> Evaluation:
        if mode not in {"full", "union", "demand"}:
            raise ValueError(f"unknown cache mode: {mode!r}")
        length = _series_length(margins)
        scores = np.full(self.n_classes, -np.inf)
        origins: list[int | None] = []
        legality: list[int] = []

        shared = _Evaluator(margins, precompute=(mode == "full"))
        shared.prepare([r for r in self.roots if r is not None])
        window_total = 0
        body_total = 0
        for c, root in enumerate(self.roots):
            if root is None:
                origins.append(None)
                legality.append(0)
                continue
            evaluator = shared if mode in {"full", "union"} else _Evaluator(margins, False)
            candidates = legal_origins(root, length)
            legality.append(len(candidates))
            if len(candidates) == 0:
                scores[c], origin = -np.inf, None
            else:
                values = evaluator.root_scores(root, candidates)
                best = int(np.argmax(values))
                scores[c], origin = float(values[best]), int(candidates[best])
            origins.append(origin)
            window_total += evaluator.window_values
            body_total += evaluator.body_values

        return Evaluation(
            scores=scores,
            origins=origins,
            legality=legality,
            window_values=window_total,
            body_values=body_total,
        )

    def predict(self, margins: dict[str, np.ndarray], mode: str = "union"):
        """First class in stored order with the maximal positive score, else abstain."""
        evaluation = self.evaluate(margins, mode=mode)
        best = evaluation.best
        if not np.isfinite(evaluation.scores[best]) or evaluation.scores[best] <= 0:
            return -1, evaluation
        return best, evaluation
