"""Stage backend verification on REAL data.

Runs every solver stage end to end on two frozen real tasks and asserts the outcome-scored Evidence is
sane. Prints raw numbers first. This is the proof the muscle works, not a claim:

  * banking77 voice-agent tool-router (10 intents, multiclass) -- profile/audit/baseline/model_scan/
    acquire/expand/certify, must CERTIFY on the locked test;
  * a deliberately uncalibrated tabular task -- the acquire stage must REFUSE on the ECE gate (the
    calibration finding, enforced).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from vectorforge import stages, science

BANK = Path("/Users/abdullahalghamdi/vectorforge-harnesses/tool-router/data")


def loadj(n):
    return [json.loads(l) for l in (BANK / f"{n}.jsonl").read_text().splitlines() if l.strip()]


def banking_splits():
    raw = loadj("train") + loadj("validation") + loadj("test")
    labels = sorted({r["target"] for r in raw})
    train, val, test, _ = science.make_splits(raw, seed=0)
    return train, val, test, labels


def synth_uncalibrated_tabular(n=1500, n_features=40, seed=1):
    """A high-dimensional, (near-)signal-free tabular task. A linear probe trained on a small seed of this
    40-dim space OVERFITS: it separates the training sample by chance and reports near-certain confidence,
    while held-out accuracy is ~chance. That confidence/accuracy gap is genuine miscalibration (large ECE),
    which is exactly the regime where uncertainty sampling selects noise. The acquire ECE gate must REFUSE.
    (Symmetric label noise does NOT do this -- it keeps a model honestly low-confidence; overfitting does.)
    Labels are 'neg'/'pos' and ids carry no label substring so the leakage audit stays clean."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        f = {f"x{j}": float(rng.normal()) for j in range(n_features)}
        y = "pos" if rng.random() < 0.5 else "neg"     # essentially no learnable signal
        rows.append({"id": f"row_{chr(97 + i % 26)}{i // 26}", "features": f, "target": y})
    train, val, test, _ = science.make_splits(rows, seed=seed, text_key="text", target_key="target")
    return train, val, test, ["neg", "pos"]


def xor_task(n=2400, n_noise=4, seed=2):
    """The canonical XOR task: y = bit(x0) XOR bit(x1), no single feature predicts the label. Linear/NB
    learners and shallow stumps cannot solve it; gradient-boosted trees solve it natively. This is the
    masked-execution case the product brain's representation axis fails on -- the solver's model_scan
    must CERTIFY it. Plus noise features so it is not a trivial 2-column problem."""
    rng = np.random.default_rng(seed)
    rows = []
    # no id field: science.audit's label_token_in_text_or_id gate substring-matches label values anywhere
    # in id/text, so any id that happens to contain a label string (a digit for "0"/"1", or a letter combo
    # spelling "odd") false-flags leakage. That over-trigger is flagged to Codex; here we just omit ids.
    for i in range(n):
        b0, b1 = int(rng.random() < 0.5), int(rng.random() < 0.5)
        f = {"x0": b0 + 0.05 * float(rng.normal()), "x1": b1 + 0.05 * float(rng.normal())}
        for j in range(n_noise):
            f[f"z{j}"] = float(rng.normal())
        rows.append({"features": f, "target": "odd" if (b0 ^ b1) else "even"})
    train, val, test, _ = science.make_splits(rows, seed=seed, text_key="text", target_key="target")
    return train, val, test, ["even", "odd"]


def show(ev):
    print(f"  [{ev['operation']:>10}] decision={ev['decision']:<22} outcome={ev['outcome']:<10} "
          f"{ev['summary']}")
    return ev


def main():
    checks = []

    print("=== TASK 1: banking77 voice-agent tool-router (multiclass) ===")
    tr, va, te, labels = banking_splits()
    metric = "accuracy"
    p = show(stages.stage_profile("text", tr, va, te, labels, metric))
    a = show(stages.stage_audit("text", tr, va, te, labels, metric, min_heldout_n=200))
    b = show(stages.stage_baseline("text", tr, va, te, labels, metric))
    s = show(stages.stage_model_scan("text", tr, va, te, labels, metric,
                                     baseline_val=b["evidence"]["val_metric"]))
    acq = show(stages.stage_acquire("text", tr, va, te, labels, metric))
    ex = show(stages.stage_expand("text", tr, va, te, labels, metric,
                                  baseline_val=b["evidence"]["val_metric"]))
    cert = show(stages.stage_certify("text", tr, va, te, labels, metric,
                                     winner=s["evidence"]["winner"], winner_cfg=s["evidence"]["winner_cfg"],
                                     threshold=0.80, alpha=0.05, max_latency_ms=50.0, min_heldout_n=200))
    print(f"    certificate: {metric}={cert['observed']} lb={cert['lower_bound']} "
          f"p={cert['p_value']} n={cert['n']} latency={cert['latency_ms_p95']}ms")

    checks.append(("profile PASS", p["outcome"] == "PASS"))
    checks.append(("audit clean (no leakage on a real task)", a["leakage_passed"] is True))
    checks.append(("baseline set a reference", b["observed"] is not None))
    checks.append(("model_scan produced a winner spec", bool(s["evidence"]["winner"])))
    checks.append(("model_scan leaderboard has >=4 strong families",
                   len(s["evidence"]["leaderboard"]) >= 4))
    checks.append(("acquire returned a measured verdict",
                   acq["decision"] in {"acquire_by_uncertainty", "no_acquisition_value",
                                       "refuse_acquisition", "not_applicable"}))
    checks.append(("expand measured (not just claimed)", ex["observed"] is not None))
    checks.append(("CERTIFIED on locked test", cert["outcome"] == "CERTIFIED"))
    checks.append(("certify lower bound clears threshold", cert["lower_bound"] > 0.80))
    checks.append(("certify leakage clean", cert["leakage_passed"] is True))

    print("\n=== TASK 2: noisy uncalibrated tabular (the ECE acquisition gate) ===")
    tr2, va2, te2, lab2 = synth_uncalibrated_tabular()
    a2 = show(stages.stage_audit("tabular", tr2, va2, te2, lab2, "accuracy", min_heldout_n=200))
    acq2 = show(stages.stage_acquire("tabular", tr2, va2, te2, lab2, "accuracy"))
    print(f"    measured ECE={acq2['evidence'].get('ece')} gate={stages.ECE_ACQUIRE_MAX}")
    checks.append(("ECE gate REFUSES acquisition on the uncalibrated task",
                   acq2["decision"] == "refuse_acquisition" and acq2["outcome"] == "REFUSED"))
    checks.append(("refusal cites the ECE measurement",
                   "ece" in acq2["evidence"] and acq2["evidence"]["ece"] > stages.ECE_ACQUIRE_MAX))

    print("\n=== TASK 3: XOR (the masked-execution case the product brain fails) ===")
    tr3, va3, te3, lab3 = xor_task()
    b3 = show(stages.stage_baseline("tabular", tr3, va3, te3, lab3, "accuracy"))
    s3 = show(stages.stage_model_scan("tabular", tr3, va3, te3, lab3, "accuracy",
                                      baseline_val=b3["observed"]))
    c3 = show(stages.stage_certify("tabular", tr3, va3, te3, lab3, "accuracy",
                                   winner=s3["evidence"]["winner"], winner_cfg=s3["evidence"]["winner_cfg"],
                                   threshold=0.85, alpha=0.05, max_latency_ms=50.0, min_heldout_n=200))
    print(f"    certificate: accuracy={c3['observed']} lb={c3['lower_bound']} n={c3['n']}")
    checks.append(("model_scan winner solves XOR (val >= 0.9)", s3["observed"] >= 0.9))
    checks.append(("XOR CERTIFIED on locked test (brain's masked failure, fixed)",
                   c3["outcome"] == "CERTIFIED" and c3["lower_bound"] > 0.85))

    print()
    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  ({sum(v for _, v in checks)}/{len(checks)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
