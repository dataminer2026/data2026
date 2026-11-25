"""PCV: compositional temporal rules with shared-origin evidence replay.

Implements the method of *PCV: Compositional Temporal Rules with Reproducible
Evidence for Multivariate Time Series*: Perceive (Sect. 3.2), Compile (Sect.
3.4) and Verify (Sect. 3.5).
"""

from .compile import FitConfig, fit
from .graph import And, Atom, Evaluation, Or, Root, RuleSet, legal_origins, windows_of
from .perceive import Perceiver
from .predicates import Concept, Normalizer, gmr_max, gmr_min
from .relations import before_cap, eq_margin, le_margin, lt_margin
from .verify import (
    Certificate,
    ReplayReport,
    artifact_id,
    build_certificate,
    replay,
    tolerance_decision,
)
from .windows import WindowRef, sliding_min, window_score

__version__ = "0.1.0"

__all__ = [
    "And",
    "Atom",
    "Certificate",
    "Concept",
    "Evaluation",
    "FitConfig",
    "Normalizer",
    "Or",
    "Perceiver",
    "ReplayReport",
    "Root",
    "RuleSet",
    "WindowRef",
    "artifact_id",
    "before_cap",
    "build_certificate",
    "eq_margin",
    "fit",
    "gmr_max",
    "gmr_min",
    "le_margin",
    "legal_origins",
    "lt_margin",
    "replay",
    "sliding_min",
    "tolerance_decision",
    "window_score",
    "windows_of",
]
