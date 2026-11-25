"""Unit tests for the PCV implementation (standard library only)."""

from __future__ import annotations

import unittest

import numpy as np

import pcv
from pcv.data import device_cycle, device_rule_graphs
from pcv.graph import And, Atom, Root, RuleSet
from pcv.perceive import Perceiver
from pcv.predicates import Concept, gmr_max, gmr_min
from pcv.relations import before_cap, eq_margin, le_margin, lt_margin
from pcv.verify import build_certificate, replay, tolerance_decision
from pcv.windows import WindowRef, sliding_min, window_scores_all


def fixed_perceiver(n_channels: int = 3, threshold: float = 0.5) -> Perceiver:
    return Perceiver.fixed(n_channels, threshold)


class EndpointMargins(unittest.TestCase):
    def test_signs(self) -> None:
        self.assertGreater(lt_margin(3, 4), 0)
        self.assertLess(lt_margin(4, 4), 0)
        self.assertGreater(le_margin(4, 4), 0)
        self.assertLess(le_margin(5, 4), 0)
        self.assertGreater(eq_margin(4, 4), 0)
        self.assertLess(eq_margin(4, 5), 0)

    def test_cap_values(self) -> None:
        # Fig. 8: kappa = 10, [L, H] = [0, 24]
        self.assertAlmostEqual(before_cap(4, 0, 24, 10.0), 0.35)
        self.assertAlmostEqual(before_cap(12, 0, 24, 10.0), 1.15)

    def test_cap_binds_only_inside_bounds(self) -> None:
        self.assertLess(before_cap(30, 0, 24, 10.0), 0)
        self.assertLess(before_cap(0, 0, 24, 10.0), 0)


class WindowScores(unittest.TestCase):
    def test_sliding_min_matches_bruteforce(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=40)
        for width in (1, 2, 4, 8):
            got = sliding_min(values, width)
            for i in range(len(values)):
                if i < width - 1:
                    self.assertTrue(np.isnan(got[i]))
                else:
                    self.assertAlmostEqual(got[i], values[i - width + 1 : i + 1].min())

    def test_window_scores_match_bruteforce(self) -> None:
        rng = np.random.default_rng(1)
        margin = rng.normal(size=30)
        concept = Concept(0, "level")
        window = WindowRef(concept, 2, 5)
        scores = window_scores_all(margin, window)
        for t in range(len(margin)):
            if t + window.b < len(margin):
                self.assertAlmostEqual(scores[t], margin[t + 2 : t + 6].min())
            else:
                self.assertTrue(np.isnan(scores[t]))


class SharedOrigin(unittest.TestCase):
    """Proposition 1: max-min <= min-max, with strict separation on split cycles."""

    def test_inequality(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(200):
            atoms = rng.normal(size=(3, 12))
            self.assertLessEqual(np.max(np.min(atoms, axis=0)), np.min(np.max(atoms, axis=1)) + 1e-12)

    def test_split_cycle_is_rejected_only_by_shared_origin(self) -> None:
        perceiver = fixed_perceiver(3)
        graphs = device_rule_graphs()
        x = np.zeros((64, 3))
        x[0:4, 0] = 1.0  # A
        x[8:12, 1] = 1.0  # first B
        x[20:24, 1] = 1.0  # second B
        x[28:32, 2] = 1.0  # C
        margins = perceiver.margins(x)

        shared = RuleSet((graphs["shared"],)).evaluate(margins).scores[0]
        r1 = RuleSet((graphs["R1"],)).evaluate(margins).scores[0]
        r2 = RuleSet((graphs["R2"],)).evaluate(margins).scores[0]
        separate = min(r1, r2)

        self.assertLessEqual(shared, 0.0)
        self.assertGreater(r1, 0.0)
        self.assertGreater(r2, 0.0)
        self.assertGreater(separate, 0.0)


class CacheModes(unittest.TestCase):
    def test_scores_are_identical_across_modes(self) -> None:
        d = device_cycle(n_per_family=5, sigma=0.1, seed=3)
        perceiver = fixed_perceiver(3)
        graphs = device_rule_graphs()
        ruleset = RuleSet((graphs["shared"], graphs["R1"]))
        for x in d.X[:6]:
            margins = perceiver.margins(x)
            scores = [ruleset.evaluate(margins, mode=m).scores for m in ("full", "union", "demand")]
            for other in scores[1:]:
                np.testing.assert_allclose(scores[0], other, rtol=0, atol=1e-12)


class DeviceCycleAccuracy(unittest.TestCase):
    def test_noiseless_balanced_accuracy(self) -> None:
        d = device_cycle(n_per_family=50, sigma=0.0, seed=11)
        perceiver = fixed_perceiver(3)
        graphs = device_rule_graphs()

        def ba(fn) -> float:
            tp = tn = fp = fn_ = 0
            for x, y in zip(d.X, d.y):
                pred = fn(perceiver.margins(x))
                tp += int(pred and y == 1)
                tn += int((not pred) and y == 0)
                fp += int(pred and y == 0)
                fn_ += int((not pred) and y == 1)
            return 0.5 * (tp / (tp + fn_) + tn / (tn + fp))

        single = lambda m: RuleSet((graphs["R1"],)).evaluate(m).scores[0] > 0
        shared = lambda m: RuleSet((graphs["shared"],)).evaluate(m).scores[0] > 0
        separate = lambda m: min(
            RuleSet((graphs["R1"],)).evaluate(m).scores[0],
            RuleSet((graphs["R2"],)).evaluate(m).scores[0],
        ) > 0

        self.assertAlmostEqual(ba(single), 2 / 3, places=6)
        self.assertAlmostEqual(ba(separate), 5 / 6, places=6)
        self.assertAlmostEqual(ba(shared), 1.0, places=6)


class Certificates(unittest.TestCase):
    def test_replay_is_exact(self) -> None:
        d = device_cycle(n_per_family=20, sigma=0.0, seed=5)
        perceiver = fixed_perceiver(3)
        graphs = device_rule_graphs()
        ruleset = RuleSet((graphs["shared"],))
        checked = 0
        for x in d.X:
            margins = perceiver.margins(x)
            predicted, evaluation = ruleset.predict(margins)
            certificate = build_certificate(
                ruleset, perceiver, margins, predicted if predicted >= 0 else -1, evaluation.origins[predicted] if predicted >= 0 else None
            )
            report = replay(ruleset, perceiver, x, certificate, tol=0.0)
            self.assertTrue(report.ok, report.notes)
            checked += 1
        self.assertGreater(checked, 0)

    def test_certificate_records_shared_window_once(self) -> None:
        d = device_cycle(n_per_family=5, sigma=0.0, seed=7)
        perceiver = fixed_perceiver(3)
        ruleset = RuleSet((device_rule_graphs()["shared"],))
        x = d.X[np.argmax(d.y)]
        margins = perceiver.margins(x)
        predicted, evaluation = ruleset.predict(margins)
        certificate = build_certificate(ruleset, perceiver, margins, predicted, evaluation.origins[predicted])
        keys = [w.key for w in certificate.windows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 3)  # A, B, C

    def test_limiter_is_the_binding_operand(self) -> None:
        d = device_cycle(n_per_family=10, sigma=0.3, seed=9)
        perceiver = fixed_perceiver(3)
        ruleset = RuleSet((device_rule_graphs()["shared"],))
        for x in d.X:
            margins = perceiver.margins(x)
            predicted, evaluation = ruleset.predict(margins)
            if predicted < 0:
                continue
            certificate = build_certificate(ruleset, perceiver, margins, predicted, evaluation.origins[predicted])
            for relation in certificate.relations:
                self.assertAlmostEqual(
                    relation.margin,
                    min(
                        [w.minimum for w in certificate.windows if w.key == relation.left][0],
                        [w.minimum for w in certificate.windows if w.key == relation.right][0],
                        relation.bound,
                    ),
                )


class Relaxation(unittest.TestCase):
    def test_gmr_preserves_sign(self) -> None:
        rng = np.random.default_rng(4)
        for _ in range(200):
            z = rng.normal(size=6)
            self.assertGreater(gmr_max(z, 8.0), 0) if z.max() > 0 else self.assertLessEqual(gmr_max(z, 8.0), 0)
            self.assertEqual(gmr_min(z, 8.0) > 0, z.min() > 0)


class ToleranceDecision(unittest.TestCase):
    def test_rules(self) -> None:
        self.assertEqual(tolerance_decision(np.array([0.5, 0.1]), 0.1), ("class", 0))
        self.assertEqual(tolerance_decision(np.array([0.5, 0.35]), 0.1)[0], "undetermined")
        self.assertEqual(tolerance_decision(np.array([-0.5, -0.9]), 0.1), ("abstain", None))


class UEALoader(unittest.TestCase):
    def test_reads_ts_file(self) -> None:
        import tempfile
        from pathlib import Path

        from pcv.data import load_ts

        text = "\n".join(
            [
                "@problemName Test",
                "@dimension 2",
                "@classLabel true 0 1",
                "@data",
                "0:1.0,2.0,3.0:4.0,5.0,6.0",
                "1:0.0,0.5,1.0:1.0,1.5,2.0",
            ]
        )
        path = Path(tempfile.gettempdir()) / "pcv_test.ts"
        path.write_text(text, encoding="utf-8")
        X, y = load_ts(path)
        self.assertEqual(X.shape, (2, 3, 2))
        self.assertEqual(list(y), ["0", "1"])
        self.assertAlmostEqual(X[0, 1, 0], 2.0)
        self.assertAlmostEqual(X[1, 2, 1], 2.0)


class Training(unittest.TestCase):
    def test_fit_recovers_one_relation_per_class(self) -> None:
        rng = np.random.default_rng(0)
        X, y = [], []
        for label, (first, second) in enumerate([(0, 1), (1, 2), (2, 0)]):
            for _ in range(30):
                start = int(rng.integers(0, 48))
                x = np.zeros((64, 3))
                x[start : start + 4, first] = 1.0
                x[start + 8 : start + 12, second] = 1.0
                X.append(x)
                y.append(label)
        X = np.stack(X)
        y = np.array(y)
        ruleset, perceiver, diagnostics = pcv.fit(
            X,
            y,
            pcv.FitConfig(transforms=("level",), widths=(4,), gaps=(4,), max_atoms=2, pool=12, beam=4),
        )
        correct = sum(ruleset.predict(perceiver.margins(x))[0] == int(label) for x, label in zip(X, y))
        self.assertGreaterEqual(correct / len(y), 0.95)
        self.assertTrue(all(len(atoms) >= 1 for atoms in diagnostics["atoms"]))


if __name__ == "__main__":
    unittest.main()
