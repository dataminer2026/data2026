"""Device-cycle composition benchmark (PCV, Sect. 4.9).

Reproduces the noiseless row of Sect. 4.9 exactly
(66.67 / 66.67 / 83.33 / 100.00) and the noise sweep of Fig. 9.

    python examples/device_cycle.py
    python examples/device_cycle.py --series 50      # quick run
"""

from __future__ import annotations

import argparse
import statistics

from pcv.data import device_cycle, device_rule_graphs
from pcv.graph import RuleSet
from pcv.perceive import Perceiver

CONFIGS = ("R1", "R2", "separate", "shared")


def predict(kind: str, margins: dict, graphs) -> bool:
    if kind == "shared":
        return RuleSet((graphs["shared"],)).evaluate(margins).scores[0] > 0
    if kind == "separate":
        first = RuleSet((graphs["R1"],)).evaluate(margins).scores[0]
        second = RuleSet((graphs["R2"],)).evaluate(margins).scores[0]
        return min(first, second) > 0
    return RuleSet((graphs[kind],)).evaluate(margins).scores[0] > 0


def balanced_accuracy(kind: str, dataset, perceiver, graphs) -> float:
    tp = tn = fp = fn = 0
    for x, y in zip(dataset.X, dataset.y):
        fired = predict(kind, perceiver.margins(x), graphs)
        tp += int(fired and y == 1)
        tn += int((not fired) and y == 0)
        fp += int(fired and y == 0)
        fn += int((not fired) and y == 1)
    return 100.0 * 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=int, default=200, help="series per family")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()

    perceiver = Perceiver.fixed(3, threshold=0.5)
    graphs = device_rule_graphs()

    print(f"device cycle: {3} channels, 64 samples, {args.series} series per family")
    print("\nnoiseless balanced accuracy (%)")
    clean = device_cycle(args.series, sigma=0.0, seed=args.seeds[0])
    for kind in CONFIGS:
        print(f"  {kind:9s} {balanced_accuracy(kind, clean, perceiver, graphs):6.2f}")

    print("\nnoise sweep (balanced accuracy, mean over seeds)")
    header = "  sigma  " + "".join(f"{k:>10s}" for k in CONFIGS)
    print(header)
    collected: dict[float, dict[str, list[float]]] = {}
    for sigma in (0.0, 0.1, 0.25, 0.5, 0.8):
        rows = {k: [] for k in CONFIGS}
        for seed in args.seeds:
            dataset = device_cycle(args.series, sigma=sigma, seed=seed)
            for kind in CONFIGS:
                rows[kind].append(balanced_accuracy(kind, dataset, perceiver, graphs))
        collected[sigma] = rows
        cells = "".join(f"{statistics.mean(rows[k]):10.2f}" for k in CONFIGS)
        print(f"  {sigma:<6.2f}" + cells)

    print("\nsigma = 0.25 (sample SD over seeds)")
    for kind in CONFIGS:
        values = collected[0.25][kind]
        print(f"  {kind:9s} {statistics.mean(values):6.2f}  (SD {statistics.stdev(values):.2f})")


if __name__ == "__main__":
    main()
