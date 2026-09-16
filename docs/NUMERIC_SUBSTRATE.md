# The Numerical Substrate - the LLM↔numbers firewall

**One typed boundary between the LLM's words and every decision-bearing number.** `vfplatform/numeric_substrate.py`
formalizes a single, falsifiable thesis: an LLM is a *language* model - it pattern-matches tokens and is
unreliable at the exact arithmetic, statistics, and bookkeeping ML rigor depends on (effect sizes, confidence
bounds, multiplicity correction, power, calibration). So across this system the LLM is confined to the one
thing it is good at - proposing **structure** (which backbone, which motif, which authored code, which axis to
escalate) - and is **structurally forbidden from producing any number that affects a decision**.

The substrate is the *single source of numerical truth*. It owns no formulas of its own: every operation
forwards to the **frozen** core (`vectorforge.science` - Clopper-Pearson, calibration, metrics;
`vfplatform.battery` - McNemar, BH-FDR; `vfplatform.power` - power / required-n). The frozen Tier-3 certifier
(`science.py b564fba2` / `sealed.py 30ad6245`) remains the sole promoter; the substrate is the disciplined
accountant that sits **between** the LLM's structural proposals and those frozen primitives.

## The rule, made enforceable

Every number has a provenance. There are exactly two kinds:

- **Trusted** (`source ∈ {computed, frozen}`) - produced *here*, by a frozen-backed producer. May decide.
- **Untrusted** (`source ∈ {llm, user}`) - an *emitted hint*. Inadmissible until `recompute()` re-derives it
  from raw data and finds it `verified` (agrees within tolerance). On disagreement the substrate's value wins
  and the hint is marked `contradicted`; with no way to recompute it is `unverifiable`.

```python
ns = NumericSubstrate()

# PRODUCE (trusted, frozen-backed): the bound the certifier uses to clear theta
lb = ns.accuracy_lower_bound(correct, alpha=0.05)     # == science.clopper_pearson_lower(k, n, 0.05)
ns.decide(lb)                                         # -> lb.value (allowed)

# POLICE (untrusted hint): an LLM claims a big lift on a TIED model
hint = ns.claim_llm("lift", 0.15)
ns.recompute(hint, lambda: ns.paired_lift(cand, base).value)   # recomputes ~0.0
assert hint.verdict == "contradicted"
ns.decide(hint)                                       # -> raises NumberLeak  (the firewall trips)
```

`decide()` is the chokepoint that makes *"no LLM number decides"* a hard, testable property:

| claim state | `decide()` returns |
|---|---|
| trusted (computed/frozen) | its value |
| untrusted, **verified** by recompute | the **substrate's recomputed value** - never the hint |
| untrusted, contradicted / unverifiable / pending | **raises `NumberLeak`** |

`audit()` returns the full ledger; `clean` is `True` iff no untrusted number was left admissible. Embed it in a
certificate as machine-checkable proof that no LLM-emitted number leaked into a decision.

## The structural half - the recipe-number guard

An LLM (or template) proposes a *recipe*, which carries numeric genes (`lr`, `weight_decay`, `epochs`).
`guard_recipe_numbers(recipe)` clamps those to the audited space (`lr ∈ [1e-5, 1e-1]`, `weight_decay ∈
[1e-6, 1e-2]`, `epochs ∈ [3, 40]`) and returns a **new** recipe (the frozen dataclass is never mutated), so a
proposal can never smuggle in an out-of-range or decision-bearing number.

```python
out, info = guard_recipe_numbers(Recipe(backbone="raw", adaptation="linear_probe", head="linear",
                                        lr=10.0, epochs=999))
assert out.lr == 0.1 and out.epochs == 40
assert info["clamped"]["lr"] == (10.0, 0.1)
```

## Where it is wired

The substrate is wired into the autonomous front door (`vfplatform/goal_solver.py`), **not** into the frozen
certify loop - the regenerative loop and the Tier-3 certifier are untouched. After a goal is certified, the
front door re-derives the certificate's headline bounds through the substrate (`single_source_of_truth`) and
runs a live adversarial self-test (`firewall_selftest`): an LLM asserts an incorrect pooled accuracy, the
substrate recomputes the true value, and `decide()` refuses it. See `docs/GOAL_SOLVER.md`.

## What this is NOT

Not a new statistical core. It owns no formulas. The lock tests assert **byte-identical** agreement with the
frozen primitives, so the substrate can never silently diverge into a second, weaker source of truth.

## Honesty locks

`tests/test_numeric_substrate.py` (18 hermetic tests, offline, deterministic) locks:
- **single source of truth** - every producer is byte-identical to calling the frozen primitive directly
  (Clopper-Pearson, McNemar, BH-FDR, power, ECE, metrics, paired lift);
- **no LLM number decides** - a contradicted hint trips `NumberLeak`; a verified hint returns the *substrate's*
  recomputed value, never the hint; an unverifiable hint is refused;
- **auditability** - `audit()` flags every untrusted number and is `clean` only when none could decide;
- **recipe-number guard** - out-of-range genes are clamped, and the frozen input recipe is never mutated.

Frozen hashes are byte-identical before and after: `science.py b564fba2`, `sealed.py 30ad6245`.
