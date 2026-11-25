# pcv

PCV classifies multivariate time series with rules over temporal relations. Each
decision comes with a certificate that another program can replay from the raw
samples, so the explanation can be checked without trusting the training code.

The paper is *PCV: Compositional Temporal Rules with Reproducible Evidence for
Multivariate Time Series Classification*.

## How it works

The method has three steps, one module each.

**Perceive.** Each signal feature is turned into a signed predicate margin

```
r_m(t) = (u_m - mu_m) / s_m - theta_m
```

where `u_m` is the transformed signal value, `mu_m` and `s_m` the training
median and IQR, and `theta_m` the threshold. The normalizer and the threshold
travel with the rule, because a margin without its unit is not reproducible.

**Compile.** A rule is a graph of relations (`Before`, `Meets`, `Overlaps`)
between windows of predicates. Windows are identified by concept and offsets,
which lets two relations share the same window instead of each carrying a copy.

**Verify.** Evaluation returns a class or abstains. It also records the origin,
the windows with their minima and limiting samples, the gap and cap of each
relation, and which operand limits the score. `replay` recomputes all of it from
raw samples through exported primitives, so changing one sample makes the
certificate fail.

## Install

```bash
pip install -e .
```

The package depends on `numpy` only. The tests use the standard library.

```bash
PYTHONPATH=. python examples/device_cycle.py --series 50
```

## Usage

```python
import pcv
from pcv.data import device_cycle, device_rule_graphs

perceiver = pcv.Perceiver.fixed(3, threshold=0.5)   # r(t) = x(t) - 0.5
ruleset = pcv.RuleSet((device_rule_graphs()["shared"],))

x = device_cycle(n_per_family=5, sigma=0.0, seed=11).X[0]
margins = perceiver.margins(x)
predicted, evaluation = ruleset.predict(margins)

certificate = pcv.build_certificate(
    ruleset, perceiver, margins, predicted, evaluation.origins[predicted]
)
report = pcv.replay(ruleset, perceiver, x, certificate, tol=0.0)
assert report.ok
```

Replaying an edited series shows the check has teeth:

```bash
PYTHONPATH=. python examples/replay_demo.py
```

```
replay (exact arithmetic): ok=True  max deviation 0.0e+00
replay after edit:        ok=False  window ch0:level[0,3] deviates by 1.000e+00
```

## Source layout

```
pcv/
  transforms.py   level, rising/falling difference, RMS energy (left replication)
  predicates.py   Eq. (1) margins, median/IQR normalizer, GMR max/min (Eq. 7)
  windows.py      Win(m,[a,b]) and q_e(t) (Eq. 2), monotone-deque sliding minimum
  relations.py    endpoint margins, geometric cap (Eq. 3), Before/Meets/Overlaps
  graph.py        interned rule graph, shared-origin evaluation, Union/Demand/Full caches
  perceive.py     step 1: concepts, normalization, thresholds, JSON export
  compile.py      step 2: candidate atoms, beam search, threshold projection
  verify.py       step 3: certificates, replay, Theorem 1 tolerance rule
  data.py         device-cycle and gap benchmarks, minimal UEA .ts loader
docs/EQUATIONS.md  paper equation to function
```

## What is reproduced, and what is not

The controlled experiment in Sect. 4.9 is reproduced, because the paper states
every parameter it needs:

| Device cycle | Paper | This repository |
|---|---|---|
| noiseless balanced accuracy, `R1` / `R2` / separate / shared | 66.67 / 66.67 / 83.33 / 100.00 | identical |
| `sigma = 0.25`, shared / separate | 87.75 / 76.19 | 87.50 / 76.67 |

The second row differs in the last digits. The paper does not say how its noise
draws are made, so the sweep here uses `numpy.random.default_rng` with seeds
11, 23 and 37, and averages the three.

The ten-dataset UEA table of Sect. 5 is not reproduced. Those datasets and the
per-run records are not part of this repository, so any number produced here
would be a new experiment rather than a check of the published one.

## Tests

```bash
python -m unittest discover -s tests -v
```

Sixteen tests target the analytic claims rather than the table:

- Proposition 1, both the inequality `max_t min_k <= min_k max_t` and a split
  cycle that only the shared origin rejects
- Theorem 1, exact replay plus the `eps` tolerance rule at its boundary
- the Union, Demand and Full caches agreeing with uncached evaluation
- a certificate that records a shared window once and names the limiting operand
- the geometric cap of Eq. (3) at its extremes
- `gmr_max` and `gmr_min` preserving sign
- sliding minima and window scores against brute force
- beam search recovering one relation per class
- the UEA `.ts` reader

## Scope and known gaps

`pcv.fit` is a reference trainer. It searches structures on the hard score and
projects thresholds onto the 5th to 95th percentile of the training data, but it
does not implement the alternating soft/hard optimisation of Eq. (8). The
controlled experiments use fixed rules and do not call it.

`Perceiver.fit` follows Eq. (1) with the training median and IQR. For
piecewise-constant signals the IQR collapses to the `1e-6` floor, which is why
the controlled experiments use `Perceiver.fixed` instead.

A class whose evidence is the absence of a pattern gets a `None` root. A
one-vs-rest conjunction cannot express it, and a root that never fires is safer
than an arbitrary one.

The `.ts` reader handles the standard format; no datasets are bundled.

## Citation

The manuscript is under review. Until it appears, cite the repository:

```
@misc{pcv2026,
  title  = {pcv: compositional temporal rules with reproducible evidence},
  year   = {2026},
  note   = {https://github.com/<owner>/pcv}
}
```

## License

MIT. See `LICENSE`.
