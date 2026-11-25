"""Evidence certificate and independent replay (PCV, Sect. 3.5 / Theorem 1).

    python examples/replay_demo.py
"""

from __future__ import annotations

import numpy as np

from pcv.data import device_cycle, device_rule_graphs
from pcv.graph import RuleSet
from pcv.perceive import Perceiver
from pcv.verify import build_certificate, replay


def main() -> None:
    dataset = device_cycle(n_per_family=20, sigma=0.0, seed=11)
    perceiver = Perceiver.fixed(3, threshold=0.5)
    ruleset = RuleSet((device_rule_graphs()["shared"],))

    x = dataset.X[np.argmax(dataset.y)]
    margins = perceiver.margins(x)
    predicted, evaluation = ruleset.predict(margins)
    certificate = build_certificate(ruleset, perceiver, margins, predicted, evaluation.origins[predicted])

    print(f"decision      class {certificate.class_index}, score {certificate.score:+.3f}, origin {certificate.origin}")
    print(f"artifact      {certificate.artifact[:12]}")
    print("windows")
    for window in certificate.windows:
        print(
            f"  {window.concept}  offsets [{window.a},{window.b}]  samples [{window.start},{window.stop}]"
            f"  min {window.minimum:+.3f} at sample {window.limiting_sample}"
        )
    print("relations")
    for relation in certificate.relations:
        gap = "-" if relation.gap is None else str(relation.gap)
        print(
            f"  {relation.kind:8s} gap {gap:>2s}  bound {relation.bound_kind}={relation.bound:+.3f}"
            f"  margin {relation.margin:+.3f}  limited by {relation.limiter}"
        )

    report = replay(ruleset, perceiver, x, certificate, tol=0.0)
    print(f"\nreplay (exact arithmetic): ok={report.ok}  max deviation {report.max_deviation:.1e}")

    tampered = x.copy()
    tampered[certificate.windows[0].start, 0] -= 1.0
    broken = replay(ruleset, perceiver, tampered, certificate, tol=1e-9)
    print(f"replay after edit:        ok={broken.ok}  {broken.notes[0] if broken.notes else ''}")


if __name__ == "__main__":
    main()
