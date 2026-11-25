"""Step 3, Verify: evidence certificates and independent replay (PCV, Sect. 3.5).

A certificate records the selected origin, every window used by the selected
branch, the endpoint check of each visited relation, and the limiting operand.
Replay recomputes all of it from the **exported primitives** (concept, offsets,
transform, centre, scale, threshold) rather than from training-time caches, so
that agreement is a real check and not a tautology.

Traversal follows Sect. 3.5: a conjunction visits both children, a disjunction
visits only the first maximising child in canonical order, and shared windows
are stored once by concept and offsets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .graph import And, Atom, Or, RuleSet, node_key
from .perceive import Perceiver

__all__ = [
    "WindowEvidence",
    "RelationEvidence",
    "Certificate",
    "ReplayReport",
    "build_certificate",
    "replay",
    "tolerance_decision",
    "artifact_id",
]


def _digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def artifact_id(ruleset: RuleSet, perceiver: Perceiver) -> str:
    return _digest(
        {
            "rules": [(node_key(r.body), r.lo, r.hi) for r in ruleset.roots],
            "perceiver": perceiver.to_dict(),
        }
    )


@dataclass(frozen=True)
class WindowEvidence:
    concept: str
    a: int
    b: int
    start: int
    stop: int
    minimum: float
    limiting_sample: int

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.concept, self.a, self.b)


@dataclass(frozen=True)
class RelationEvidence:
    node: str
    kind: str
    left: tuple[str, int, int]
    right: tuple[str, int, int]
    gap: int | None
    bound_kind: str
    bound: float
    margin: float
    limiter: str


@dataclass(frozen=True)
class Certificate:
    artifact: str
    class_index: int
    score: float
    origin: int | None
    scores: tuple[float, ...]
    windows: tuple[WindowEvidence, ...]
    relations: tuple[RelationEvidence, ...]
    limiter: str | None
    abstained: bool = False


@dataclass
class ReplayReport:
    class_matches: bool = True
    origin_matches: bool = True
    score_matches: bool = True
    windows_match: bool = True
    relations_match: bool = True
    max_deviation: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(
            [
                self.class_matches,
                self.origin_matches,
                self.score_matches,
                self.windows_match,
                self.relations_match,
            ]
        )


def _window_evidence(margins, concept: str, a: int, b: int, origin: int) -> WindowEvidence:
    series = margins[concept]
    start, stop = origin + a, origin + b
    segment = series[start : stop + 1]
    idx = int(np.argmin(segment))
    return WindowEvidence(
        concept=concept,
        a=a,
        b=b,
        start=start,
        stop=stop,
        minimum=float(segment[idx]),
        limiting_sample=start + idx,
    )


def _value(node, origin: int, margins) -> float:
    """Body value at one origin, without recording evidence."""
    if isinstance(node, Atom):
        left = float(np.min(margins[node.left.concept.name][node.left.start(origin) : node.left.stop(origin) + 1]))
        right = float(np.min(margins[node.right.concept.name][node.right.start(origin) : node.right.stop(origin) + 1]))
        bound = node.cap if node.kind == "before" else node.endpoint_margin(origin)
        return float(min(left, right, bound))
    values = [_value(c, origin, margins) for c in node.children]
    return float(min(values) if isinstance(node, And) else max(values))


def _walk(node, origin: int, margins, windows: dict, relations: list) -> float:
    if isinstance(node, Atom):
        left = _window_evidence(margins, node.left.concept.name, node.left.a, node.left.b, origin)
        right = _window_evidence(margins, node.right.concept.name, node.right.a, node.right.b, origin)
        windows.setdefault(left.key, left)
        windows.setdefault(right.key, right)
        if node.kind == "before":
            bound_kind, bound = "cap", float(node.cap)
        else:
            bound_kind, bound = "endpoint", float(node.endpoint_margin(origin))
        candidates = {"left": left.minimum, "right": right.minimum, bound_kind: bound}
        limiter = min(candidates, key=lambda k: (candidates[k], k))
        relation = RelationEvidence(
            node=_digest(node_key(node))[:8],
            kind=node.kind,
            left=left.key,
            right=right.key,
            gap=node.gap,
            bound_kind=bound_kind,
            bound=bound,
            margin=float(min(candidates.values())),
            limiter=limiter,
        )
        relations.append(relation)
        return relation.margin
    if isinstance(node, And):
        values = [_walk(c, origin, margins, windows, relations) for c in node.children]
        return float(min(values))
    if isinstance(node, Or):
        values = [_value(c, origin, margins) for c in node.children]
        chosen = int(np.argmax(values))  # first maximiser in canonical order
        return float(_walk(node.children[chosen], origin, margins, windows, relations))
    raise TypeError(f"not a body node: {type(node)!r}")


def build_certificate(
    ruleset: RuleSet,
    perceiver: Perceiver,
    margins: dict[str, np.ndarray],
    class_index: int,
    origin: int | None,
) -> Certificate:
    evaluation = ruleset.evaluate(margins)
    if class_index < 0 or origin is None:
        return Certificate(
            artifact=artifact_id(ruleset, perceiver),
            class_index=-1,
            score=float(np.max(evaluation.scores)),
            origin=None,
            scores=tuple(float(s) for s in evaluation.scores),
            windows=(),
            relations=(),
            limiter=None,
            abstained=True,
        )

    root = ruleset.roots[class_index]
    windows: dict[tuple, WindowEvidence] = {}
    relations: list[RelationEvidence] = []
    score = _walk(root.body, origin, margins, windows, relations)

    limiter = None
    if isinstance(root.body, And):
        values = [_value(c, origin, margins) for c in root.body.children]
        limiter = _digest(node_key(root.body.children[int(np.argmin(values))]))[:8]

    return Certificate(
        artifact=artifact_id(ruleset, perceiver),
        class_index=int(class_index),
        score=float(score),
        origin=int(origin),
        scores=tuple(float(s) for s in evaluation.scores),
        windows=tuple(sorted(windows.values(), key=lambda w: w.key)),
        relations=tuple(relations),
        limiter=limiter,
        abstained=False,
    )


def replay(
    ruleset: RuleSet,
    perceiver: Perceiver,
    x: np.ndarray,
    certificate: Certificate,
    tol: float = 0.0,
) -> ReplayReport:
    """Recompute a certificate from raw samples and compare.

    ``tol=0`` is the exact-arithmetic check of Theorem 1; ``tol>0`` allows for a
    floating-point error budget.
    """
    report = ReplayReport()
    margins = perceiver.margins(x)

    minima: dict[tuple, float] = {}
    for ev in certificate.windows:
        segment = margins[ev.concept][ev.start : ev.stop + 1]
        recomputed = float(np.min(segment))
        minima[ev.key] = recomputed
        deviation = abs(recomputed - ev.minimum)
        report.max_deviation = max(report.max_deviation, deviation)
        if deviation > tol:
            report.windows_match = False
            report.notes.append(f"window {ev.concept}[{ev.a},{ev.b}] deviates by {deviation:.3e}")

    for rel in certificate.relations:
        recomputed = float(min(minima[rel.left], minima[rel.right], rel.bound))
        deviation = abs(recomputed - rel.margin)
        report.max_deviation = max(report.max_deviation, deviation)
        if deviation > tol:
            report.relations_match = False
            report.notes.append(f"relation {rel.node} deviates by {deviation:.3e}")

    if not certificate.abstained and certificate.origin is not None:
        windows: dict = {}
        relations: list = []
        recomputed = _walk(ruleset.roots[certificate.class_index].body, certificate.origin, margins, windows, relations)
        deviation = abs(recomputed - certificate.score)
        report.max_deviation = max(report.max_deviation, deviation)
        report.score_matches = deviation <= tol
        if not report.score_matches:
            report.notes.append(f"root score deviates by {deviation:.3e}")

    predicted, evaluation = ruleset.predict(margins)
    report.class_matches = predicted == certificate.class_index
    if not report.class_matches:
        report.notes.append(f"class {predicted} != certified {certificate.class_index}")
    if certificate.origin is not None and predicted == certificate.class_index and predicted >= 0:
        report.origin_matches = int(evaluation.origins[predicted]) == certificate.origin
        if not report.origin_matches:
            report.notes.append(f"origin {evaluation.origins[predicted]} != {certificate.origin}")
    return report


def tolerance_decision(scores: np.ndarray, eps: float) -> tuple[str, int | None]:
    """Decision rule of Theorem 1 under a uniform error budget ``eps``."""
    scores = np.asarray(scores, dtype=float)
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-finite, kind="stable")
    best = int(order[0])
    top = float(finite[best])
    if not np.isfinite(top):
        return "abstain", None
    if top < -eps:
        return "abstain", None
    lead = top - float(finite[order[1]]) if len(order) > 1 else np.inf
    if top > eps and lead > 2 * eps:
        return "class", best
    return "undetermined", None
