# Equation to code

Maps each numbered equation of the PCV manuscript to the function that
implements it.

| Paper | Meaning | Code |
|---|---|---|
| Eq. (1) | `r_m(t) = (u_m - mu_m)/s_m - theta_m` | `pcv.predicates.Normalizer.margin`, `pcv.perceive.Perceiver.margin` |
| Eq. (2) | `q_e(t) = min_{j in [a,b]} r_m(t+j)` | `pcv.windows.window_score`, `window_scores_all` |
| Eq. (3) | `bR = kappa^-1 min{g-1/2, g-L+1/2, H-g+1/2}` | `pcv.relations.before_cap` |
| Eq. (4) | `v_alpha(t) = min{q_ei, q_ej, bR}` | `pcv.graph.Atom.value` |
| Eq. (5) | `Before(A,B) ^ Before(B,C)` on a shared `B` | `pcv.data.device_rule_graphs` |
| Eq. (6) | `max_t min_k <= min_k max_t` (Proposition 1) | `tests.test_pcv.SharedOrigin` |
| Eq. (7) | `Maxp`, `Minp` generalized-mean relaxation | `pcv.predicates.gmr_max`, `gmr_min` |
| Eq. (8) | training loss | not implemented, see below |
| Theorem 1 | identical decisions and certificates in exact arithmetic | `pcv.verify.replay` (`tol=0`), `tolerance_decision` |
| Sect. 3.3 | Demand / Union / Full caching | `pcv.graph.RuleSet.evaluate(mode=...)` |
| Sect. 3.5 | certificate traversal, first maximising disjunct, limiter | `pcv.verify._walk`, `build_certificate` |

## Not implemented

Eq. (8), the training objective. Training here is limited to `pcv.compile.fit`,
described in the README as a reference trainer: it searches structures on the
hard score and projects thresholds onto the 5th to 95th percentile of the
training data.

The ten-dataset UEA comparison is also absent. Those datasets and the per-run
records are not part of this repository, so re-deriving the table here would
produce different numbers instead of a reproduction.

`Perceiver.fit` follows Eq. (1) with the training median and IQR. For
piecewise-constant signals the IQR collapses, which is why the controlled
experiments use `Perceiver.fixed`.
