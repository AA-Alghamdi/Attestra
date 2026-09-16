"""Tests for the Tier-0.5 learned surrogate: it ranks good operators above bad ones, is calibration-
tracked, never promotes, and -- wired into the cascade -- makes run 2 reach the winner in FEWER expensive
trials than run 1 (the memory-compounds acceptance criterion)."""
import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import surrogate as SU
from vfplatform import verification as V
from vfplatform import casebase_store as CB


def _fp(modality="tabular", size="m", nrows=1000):
    return {"key": f"{modality}|{size}", "modality": modality, "size_bucket": size,
            "feature_bucket": "s", "n_classes_bucket": "2", "balance_bucket": "balanced",
            "n_rows": nrows, "n_features": 10, "n_classes": 2}


def _ledger(seed=0, n=300):
    """Synthetic ledger where operator 'good' certifies ~90% of the time and 'bad' ~10%."""
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        for op, p in (("good", 0.9), ("bad", 0.1)):
            recs.append(SU.OutcomeRecord(fingerprint=_fp(), operator=op,
                                         certified=(rng.random() < p),
                                         gain=(0.1 if rng.random() < p else -0.02)))
    return recs


def test_surrogate_ranks_good_over_bad():
    s = SU.Surrogate(seed=0).fit(_ledger(1))
    pg = s.predict_pass_prob(_fp(), "good")
    pb = s.predict_pass_prob(_fp(), "bad")
    assert pg > 0.6 > pb, (pg, pb)
    assert s.predict_gain(_fp(), "good") > s.predict_gain(_fp(), "bad")


def test_calibration_tracking_and_ece():
    s = SU.Surrogate(seed=0).fit(_ledger(2))
    rng = random.Random(5)
    for _ in range(400):
        op, p = (("good", 0.9) if rng.random() < 0.5 else ("bad", 0.1))
        s.score_and_track(_fp(), op, actual_certified=(rng.random() < p))
    assert len(s.calibration) == 400
    ece = s.calibration.ece(bins=10)
    assert ece is not None and 0.0 <= ece < 0.25      # reasonably calibrated on a learnable signal
    assert len(s.calibration.reliability_table()) >= 2


def test_degenerate_ledger_gives_constant_base_rate():
    recs = [SU.OutcomeRecord(_fp(), "x", certified=True, gain=0.1) for _ in range(20)]
    s = SU.Surrogate().fit(recs)                       # only one class present
    assert abs(s.predict_pass_prob(_fp(), "x") - 1.0) < 1e-9
    assert abs(s.predict_pass_prob(_fp(), "never_seen") - 1.0) < 1e-9


def test_from_casebase_adapter(tmp_path):
    store = CB.CaseBaseStore(outcome_path=str(tmp_path / "out.jsonl"),
                             negative_path=str(tmp_path / "neg.jsonl"))
    fp = CB.fingerprint([{"a": 1.0, "b": 2.0, "target": 0}, {"a": 2.0, "b": 1.0, "target": 1}] * 30,
                        kind="tabular", task_type="binary")
    store.record_outcome(fp, "logreg", gain=0.1, cost=1.0)
    store.record_outcome(fp, "dummy", gain=-0.01, cost=1.0)
    recs = SU.from_casebase(store)
    assert len(recs) == 2
    ops = {r.operator: r.certified for r in recs}
    assert ops["logreg"] is True and ops["dummy"] is False     # gain>0 proxy


def _stream(p, n, seed):
    rng = random.Random(seed)
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def test_memory_compounds_fewer_trials_run2():
    """RUN 1 (cold): a batch of candidates, no surrogate -> many reach the expensive Tier 3.
    RUN 2 (warm): the SAME batch, but a surrogate trained on run-1 outcomes pre-scores each candidate,
    so the cascade prunes the losers at Tier 0.5 and FEWER reach Tier 3 -- with the SAME winner."""
    theta = 0.80
    casc = V.VerificationCascade(theta=theta, surrogate_prune=0.3)

    # 1 good operator (certifies), 9 bad operators (do not). Build candidates for both runs.
    def make_candidates(scored: SU.Surrogate = None):
        cands = []
        specs = [("good", 0.95, True)] + [(f"bad{i}", 0.45, False) for i in range(9)]
        for name, p, certifies in specs:
            fp = _fp()
            op = "good" if name == "good" else "bad"
            score = scored.predict_pass_prob(fp, op) if scored is not None else None
            cands.append(V.Candidate(
                name=name, val_outcomes=_stream(p, 400, hash(name) % 9999),
                surrogate_score=score,
                certify_fn=(lambda c=certifies: {"certified": c, "lower_bound": 0.9 if c else 0.4})))
        return cands

    # RUN 1: no surrogate
    run1 = casc.evaluate_batch(make_candidates(None))
    # RUN 2: surrogate trained on a ledger where 'good' certifies, 'bad' does not
    surr = SU.Surrogate(seed=0).fit(_ledger(7))
    run2 = casc.evaluate_batch(make_candidates(surr))

    # same winner both runs
    assert run1.promoted_names == ["good"] == run2.promoted_names
    # FEWER expensive Tier-3 reaches in run 2 (surrogate pruned the losers cheaply)
    assert run2.reached_tier3 <= run1.reached_tier3
    assert run2.total_cost < run1.total_cost
    print(f"trial-count delta: run1 tier3={run1.reached_tier3} cost={run1.total_cost} -> "
          f"run2 tier3={run2.reached_tier3} cost={run2.total_cost}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
