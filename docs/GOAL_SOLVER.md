# The autonomous `/goal` front door

**Hand it a free-text goal and arbitrary data; walk away; get back a certified model - or an honest decline.**
`vfplatform/goal_solver.py` is the single entrypoint that closes the loop from a problem *description* to a
`GoalCertificate`. It routes through the system's existing, audited machinery - it introduces **no new
statistical primitive** and never touches the frozen certifier (`science.py b564fba2` / `sealed.py 30ad6245`).

```python
from vfplatform import goal_solver as G

cert = G.solve("classify the wine cultivar from chemical measurements", "wine")
print(cert.solved, cert.champion, cert.pooled_sealed_lb, cert.theta_floor)
# True  'raw · linear_probe · head=linear'  0.9342  0.5
```

CPU-only and deterministic given a seed when `online=False` (the default: bundled literature corpus + the
deterministic template authorer). `scripts/run_goal.py` is the CLI wrapper.

## The journey `solve()` runs

1. **acquire(data)** → a dense `(X, y, n_classes, task_hint)`. Accepts inline `(X, y)` arrays, a
   `{"X","y"}` dict, a `.npz`, a `.csv` (target = named column or last), a bundled sklearn name
   (`wine`/`digits`/`breast_cancer`/`diabetes`/`california`/`iris`/`covtype`), an OpenML-CC18 member, or an
   `openml://<id>` URI. Regression vs classification is decided from the target; string labels are encoded.
2. **infer_spec(X, y, goal)** → the certifiable spec: modality/task/metric come from the deterministic-first
   `vfplatform.problem_type`; the arena **shape** (balanced / natural-proportion / regression) and **split**
   discipline (random, or grouped/time when real `groups`/`times` are supplied) come from the data. An
   out-of-scope goal (forecasting, ranking, audio) is **declined** here - never coerced into a fake fit.
3. **_build(spec)** → a `TabularCodeArena` + a literature-grounded `RecipeGenerator` (a `LiteratureScout`
   supplies retrieved motifs/backbones with provenance) + a `RecipeResearcher`. The seed recipes pass through
   the substrate's recipe-number guard. The **competence floor** `theta` is set strictly *above* the
   meta-certifier's own trivial baseline (see below).
4. **researcher.run()** → the regenerative loop: propose recipes → screen → the frozen Tier-3 certifier is the
   sole promoter. Returns the certified champion, sealed bounds, gold confirmation, novelty/provenance.
5. **_audit_numbers(substrate, cert)** → the NumericSubstrate re-derives the certificate's headline bounds
   from the frozen core and runs the live firewall self-test (see `docs/NUMERIC_SUBSTRATE.md`).
6. **GoalCertificate** - one artifact wrapping the whole journey (JSON-serializable).

## The competence floor (why an honest theta matters)

The meta-certifier rejects a framing as *gameable* if a trivial majority-class predictor clears `theta`. It
defines "trivial" as the **train-majority class scored on the eval rows**. A naive floor of "train-majority
fraction + margin" is the *wrong quantity* - on an imbalanced set the majority class can score higher on the
eval rows than its train fraction, so the framing gets rejected. The front door therefore sets

```
theta = max(0.5, trivial_baseline_on_eval_rows + 0.03)
```

computed from the same aggregate label statistic the referee itself reads (never any model's predictions, so
nothing is snooped). The floor is then *provably* non-trivial and the framing is accepted. On a balanced set
the trivial baseline is ≈1/K, so `theta` stays `0.5` and nothing changes.

## `GoalCertificate` (headline fields)

| field | meaning |
|---|---|
| `solved` | a model is **certified above the competence floor** (`pooled_sealed_lb > theta_floor`) |
| `improved` | the loop **promoted** a champion *beyond the seed* (a real lift) |
| `declined` | the goal/data was out of supported scope (no run attempted) |
| `refused` | the meta-certifier / data-hygiene gate rejected the framing/data |
| `champion`, `champion_recipe` | the certified recipe |
| `theta_floor`, `pooled_sealed_lb` | the bar and the pooled sealed Clopper-Pearson lower bound that clears it |
| `sealed_acc`, `sealed_lb`, `gold_confirmation` | per-shard sealed accuracy / lower bounds / never-peeked gold |
| `numeric_audit` | the substrate ledger: `clean`, `firewall_held`, `single_source_of_truth`, `firewall_selftest` |
| `novelty`, `literature` | anti-menu record + champion→paper/repo/Hub provenance |
| `stop_reason`, `peeks_used` | why the loop stopped and how much sealed budget it spent |

## Verified end-to-end (offline, deterministic, CPU)

| goal / data | outcome | theta | pooled sealed LB | note |
|---|---|---|---|---|
| `classify the wine cultivar` / `wine` | **solved** | 0.500 | 0.934 | seed already competent (`improved=false`) |
| `classify iris species` / `iris` | **solved** | 0.500 | 0.804 | |
| `recognize handwritten digits` / `digits` | **solved** | 0.500 | 0.931 | |
| `detect malignant tumors` / `breast_cancer` | **solved** | 0.699 | 0.935 | imbalanced; theta clears the trivial baseline (would be `refused` with a train-only floor) |
| `classify the 3-cluster nonlinear signal` / synthetic | **solved + improved** | 0.500 | - | loop **promotes** `head=gbm` past the linear seed |
| `predict diabetes disease progression` / `diabetes` | **not solved** (honest) | 0.500 | 0.477 | tolerance hit-rate below floor - reported, not faked |
| `forecast next quarter revenue` / wine arrays | **declined** | - | - | out of scope; routed to the timeseries path, 0 peeks spent |

In every run the numeric audit is `clean=true` and `firewall_held=true` (the adversarial LLM accuracy claim is
recomputed and refused), the single-source-of-truth recomputation agrees with the reported bounds, and the
frozen hashes are byte-identical before and after.

## Honesty locks

`tests/test_goal_solver.py` (19 hermetic tests, offline, deterministic) locks acquisition (arrays/dict/npz/csv/
sklearn, label-encoding, regression detection), spec inference (multiclass/regression/imbalanced + declines for
forecasting/ranking + grouped split), the end-to-end certify, the competence-floor fix on an imbalanced set,
a real promotion past the seed, determinism, JSON round-trip, and - after a full `solve()` - that
`science.py b564fba2` / `sealed.py 30ad6245` are **byte-identical**.
