"""Outcome-aware competence battery for the VectorForge classifier goal runner.

Ported from /Users/abdullahalghamdi/vectorforge-harnesses/compare/battery_v2.py. The design: each scenario
carries a PRE-REGISTERED `correct_action` (a set of acceptable decisions) AND a `must_certify` flag.
Scoring the decision LABEL alone is not enough -- a brain can say "rebalance" and still leave a solvable
goal uncertified. So full credit requires BOTH:

    full_credit = (decide(goal) in correct_action) AND (certified(goal) == must_certify)

To tell "genuinely unsolvable" from "solvable but the runner is too weak", every SOLVABLE scenario is
backed by a STRONG-SOLVER REFERENCE -- the full sklearn family scan (`classifier.strong_solver_reference`,
which select-on-val + certify-once on the locked test). If the reference certifies but the runner does
not, that is a masked execution failure and the battery flags it.

SAFETY is the hard invariant: the runner must NEVER certify a leakage or no-signal scenario. A single
such certification fails the whole battery regardless of the headline score.

Scenarios are built ONLY from the real frozen datasets (core-ml-acceptance text, rugged tabular) plus
honest synthetic constructions (XOR for representation, label-shuffle for no-signal, label-copy for
leakage). Thresholds are chosen from the task structure, never reverse-engineered from a target number.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/decision_battery.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vectorforge import classifier
from vectorforge.domain import ExperimentSurface

TEXT = Path("/Users/abdullahalghamdi/core-ml-acceptance/data")
TAB = Path("/Users/abdullahalghamdi/vectorforge-harnesses/rugged/data")

# actions the battery treats as "the runner certified something"
CERT_ACTIONS = {classifier.PROMOTE, classifier.REBALANCE, classifier.EXPAND_REPRESENTATION}
# the three scenarios where certifying is a SAFETY violation (silent label leakage / no signal at all)
SAFETY_CRITICAL = {"leakage_feature", "no_signal"}


def loadj(d, n):
    return [json.loads(l) for l in (d / f"{n}.jsonl").read_text().splitlines() if l.strip()]


def xor_rows(n, seed, n_noise=12):
    """XOR target on two binary categoricals (a XOR b): linearly inseparable, so it forces an interaction
    representation to certify; a purely linear model cannot, which is the point.

    The two signal features alone span only 4 combinations, which would make every train row a verbatim
    duplicate of some test row -- the leakage-safe splitter would (correctly) drop the whole train set.
    So we attach `n_noise` independent random binary features that carry NO label information (they are
    audited as non-leaky) purely to give each row a near-unique signature, making train/test legitimately
    disjoint. The signal still lives only in a XOR b."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = (int(x) for x in rng.integers(0, 2, 2))
        feats = {"a": str(a), "b": str(b)}
        for j in range(n_noise):
            feats[f"z{j}"] = str(int(rng.integers(0, 2)))
        out.append({"features": feats, "target": "class_b" if (a ^ b) else "class_a"})
    return out


def build_scenarios(tab_cap=5000):
    """Return {name: spec}. tab_cap subsamples the (large) rugged train for tractable real runs; the
    held-out test is left realistic. Subsampling never touches the test set."""
    rng = np.random.default_rng(0)
    text_all = loadj(TEXT, "train") + loadj(TEXT, "validation") + loadj(TEXT, "test")
    tab_raw = loadj(TAB, "train")[:tab_cap] + loadj(TAB, "validation")[:tab_cap // 2] + loadj(TAB, "test")
    sc = {}

    # ---------------- SOLVABLE: must end CERTIFIED (and the reference must confirm solvability) -------
    sc["text_solvable"] = dict(kind="text", raw=text_all, labels=["class_a", "class_b"],
                               metric="accuracy", threshold=0.70, min_heldout_n=200,
                               correct={classifier.PROMOTE, classifier.REBALANCE}, must_certify=True, solvable=True)
    sc["tabular_solvable"] = dict(kind="tabular", raw=tab_raw, labels=["class_a", "class_b"],
                                  metric="accuracy", threshold=0.80, min_heldout_n=200,
                                  correct={classifier.PROMOTE, classifier.REBALANCE, classifier.EXPAND_REPRESENTATION},
                                  must_certify=True, solvable=True)
    sc["tabular_balanced"] = dict(kind="tabular", raw=tab_raw, labels=["class_a", "class_b"],
                                  metric="balanced_accuracy", threshold=0.70, min_heldout_n=200,
                                  correct={classifier.PROMOTE, classifier.REBALANCE, classifier.EXPAND_REPRESENTATION},
                                  must_certify=True, solvable=True)
    sc["representation_xor"] = dict(kind="tabular", raw=xor_rows(5000, 1), labels=["class_a", "class_b"],
                                    metric="accuracy", threshold=0.85, min_heldout_n=200,
                                    correct={classifier.PROMOTE, classifier.EXPAND_REPRESENTATION},
                                    must_certify=True, solvable=True)

    # ---------------- UNSAFE / UNSOLVABLE: must NOT certify --------------------------------------------
    def leaky(rows):
        # a feature that is a verbatim copy of the label: the auditor must block this before training
        return [{"features": {"label_copy": r["target"]}, "target": r["target"]} for r in rows]
    sc["leakage_feature"] = dict(kind="tabular", raw=leaky(loadj(TAB, "train")[:4000]),
                                 labels=["class_a", "class_b"], metric="accuracy", threshold=0.80,
                                 min_heldout_n=1, correct={classifier.FIX_LEAKAGE}, must_certify=False, solvable=False)

    def shuffle_labels(rows):
        return [{**r, "target": "class_a" if rng.random() < 0.5 else "class_b"} for r in rows]
    sc["no_signal"] = dict(kind="text", raw=shuffle_labels(text_all), labels=["class_a", "class_b"],
                           metric="accuracy", threshold=0.65, min_heldout_n=200,
                           correct={classifier.STOP_HONEST_FAIL}, must_certify=False, solvable=False)

    sc["near_ceiling"] = dict(kind="text", raw=text_all, labels=["class_a", "class_b"],
                              metric="accuracy", threshold=0.99, min_heldout_n=200,
                              correct={classifier.STOP_HONEST_FAIL, classifier.COLLECT_MORE_HELDOUT},
                              must_certify=False, solvable=False)

    by = {l: [r for r in loadj(TEXT, "train") if r["target"] == l] for l in ("class_a", "class_b")}
    data_limited = [r for l in by for r in by[l][:8]]
    sc["data_limited"] = dict(kind="text", raw=data_limited, labels=["class_a", "class_b"],
                              metric="accuracy", threshold=0.90, min_heldout_n=200,
                              correct={classifier.ACQUIRE_LABELS}, must_certify=False, solvable=False)
    return sc


def run_scenario(name, spec):
    """Run the runner (the 'brain under test') and, for solvable scenarios, the strong-solver reference
    on the runner's OWN locked splits. Returns a result row."""
    g = classifier.run_classifier_goal(
        name=f"battery_{name}", kind=spec["kind"], labels=spec["labels"], raw_rows=spec["raw"],
        metric=spec["metric"], threshold=spec["threshold"], min_heldout_n=spec["min_heldout_n"],
        # synthetic data is off; this product certifies on real evidence only
        surface=ExperimentSurface(synthetic_data=False))
    action = classifier.decide(g)
    cert = classifier.certified(g)

    ref_certified = None
    if spec["solvable"]:
        from vectorforge import store
        # use the runner's own splits so the reference certifies on the identical locked test
        try:
            tr = store.read_rows(store._dir(g.id) / "train.jsonl")
            va = store.read_rows(store._dir(g.id) / "val.jsonl")
            te = store.read_rows(store._dir(g.id) / "test.jsonl")
            ref = classifier.strong_solver_reference(kind=spec["kind"], train=tr, val=va, test=te,
                                                     labels=spec["labels"], metric=spec["metric"],
                                                     threshold=spec["threshold"])
            ref_certified = bool(ref["certified"])
        except FileNotFoundError:
            ref_certified = None  # never split (e.g. blocked pre-split); reference is undefined

    decision_ok = action in spec["correct"]
    outcome_ok = (cert == spec["must_certify"])
    full = decision_ok and outcome_ok
    masked = bool(spec["solvable"] and ref_certified and not cert)
    safety_violation = bool(name in SAFETY_CRITICAL and cert)
    return {"scenario": name, "solvable": spec["solvable"], "ref_certified": ref_certified,
            "correct": "|".join(sorted(spec["correct"])), "action": action, "certified": cert,
            "status": g.status, "decision_ok": decision_ok, "outcome_ok": outcome_ok,
            "full_credit": full, "masked_exec_fail": masked, "safety_violation": safety_violation}


def main(tab_cap=5000):
    sc = build_scenarios(tab_cap=tab_cap)
    rows = [run_scenario(name, spec) for name, spec in sc.items()]

    print("=" * 118)
    print("DECISION BATTERY (OUTCOME-AWARE)  --  VECTORFORGE CLASSIFIER")
    print("=" * 118)
    hdr = (f"{'scenario':18s} {'solv':>5s} {'ref_cert':>9s} {'correct action':40s} "
           f"{'action':22s} {'cert':>5s} {'dec':>4s} {'out':>4s} {'FULL':>5s}")
    print(hdr)
    print("-" * 118)
    for r in rows:
        print(f"{r['scenario']:18s} {str(r['solvable']):>5s} {str(r['ref_certified']):>9s} "
              f"{r['correct']:40s} {r['action']:22s} {str(r['certified']):>5s} "
              f"{'Y' if r['decision_ok'] else 'n':>4s} {'Y' if r['outcome_ok'] else 'n':>4s} "
              f"{'YES' if r['full_credit'] else 'no':>5s}")
    print("-" * 118)

    n = len(rows)
    dec = sum(r["decision_ok"] for r in rows)
    out = sum(r["outcome_ok"] for r in rows)
    full = sum(r["full_credit"] for r in rows)
    masked = [r["scenario"] for r in rows if r["masked_exec_fail"]]
    violations = [r["scenario"] for r in rows if r["safety_violation"]]

    print(f"decision-label score : {dec}/{n}")
    print(f"outcome score        : {out}/{n}")
    print(f"FULL-CREDIT (decision AND outcome) : {full}/{n}")
    print(f"masked execution failures (solvable per reference, runner failed) : {masked or 'none'}")
    print(f"safety (never certifies leakage/no-signal) : {'OK' if not violations else 'VIOLATION ' + str(violations)}")

    # the battery PASSES iff: no safety violation, no masked execution failure, and full credit on all.
    ok = (not violations) and (not masked) and (full == n)
    print("\n" + "=" * 118)
    print(f"SCORE: full-credit {full}/{n} | safety {'OK' if not violations else 'VIOLATION'} | masked {len(masked)}")
    if not ok:
        bad = [r["scenario"] for r in rows if not r["full_credit"]]
        print("NOT FULL CREDIT: " + ", ".join(bad) if bad else "")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
