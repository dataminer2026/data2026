"""Endpoint relations and the bounded-``Before`` cap (PCV, Sect. 3.2).

Inclusive sample windows are mapped to half-open intervals ``[a, b+1)``.  For
integer endpoints a strict inequality ``x < y`` has signed margin ``y - x - 1/2``
and equality has margin ``1/2 - |x - y|``; both are positive exactly on the
intended configuration.

Eq. (3):   bR = kappa^-1 * min{ g - 1/2, g - L + 1/2, H - g + 1/2 },  g = c - b - 1
Eq. (4):   v_alpha(t) = min{ q_ei(t), q_ej(t), bR }
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "lt_margin",
    "le_margin",
    "eq_margin",
    "before_cap",
    "relation_margin",
    "RELATIONS",
]


def lt_margin(x: int, y: int) -> float:
    """Signed margin for ``x < y`` with integer endpoints."""
    return float(y - x - 0.5)


def le_margin(x: int, y: int) -> float:
    """Signed margin for ``x <= y`` with integer endpoints."""
    return float(y - x + 0.5)


def eq_margin(x: int, y: int) -> float:
    """Signed margin for ``x == y`` with integer endpoints."""
    return float(0.5 - abs(x - y))


def before_cap(gap: int, lo: int, hi: int, kappa: float) -> float:
    """Geometric cap of Eq. (3) for an unoccupied gap of ``gap`` samples."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return min(gap - 0.5, gap - lo + 0.5, hi - gap + 0.5) / kappa


RELATIONS = ("before", "meets", "overlaps")


@dataclass(frozen=True)
class RelationResult:
    value: float
    gap: int | None
    cap: float | None


def relation_margin(
    kind: str,
    left_start: int,
    left_stop: int,
    right_start: int,
    right_stop: int,
    lo: int = 0,
    hi: int = 24,
    kappa: float = 10.0,
) -> RelationResult:
    """Endpoint margin of one relation atom at a fixed origin.

    ``before`` additionally applies the geometric cap of Eq. (3); ``meets`` and
    ``overlaps`` use the corresponding endpoint tests of the Allen family.
    """
    if kind == "before":
        gap = right_start - left_stop - 1
        cap = before_cap(gap, lo, hi, kappa)
        return RelationResult(min(lt_margin(left_stop, right_start), cap), gap, cap)
    if kind == "meets":
        return RelationResult(eq_margin(left_stop + 1, right_start), None, None)
    if kind == "overlaps":
        value = min(
            lt_margin(left_start, right_start),
            le_margin(right_start, left_stop),
            lt_margin(left_stop, right_stop),
        )
        return RelationResult(value, None, None)
    raise ValueError(f"unknown relation: {kind!r}")
