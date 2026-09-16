"""Tests for vfplatform.casebase_store: durable, warm-started proposal-policy memory.

Coverage:
  (a) fingerprint from BOTH a records list and a profile dict -> same key for the same data;
      balance buckets; modality; distance is a sensible ordering.
  (b) append-only outcome log is durable + corrupt-tolerant.
  (c) warm_start is EMPTY on a cold start and INFORMATIVE (priors improve) after simulated runs;
      nearest-fingerprint weighting prefers same-dataset history over similar-dataset history.
  (d) mine_promotion_ledger folds attributable rows into priors and SKIPS unattributable ones
      (never guesses a family/profile that the frozen ledger schema does not carry).
  (e) negative memory deprioritizes dead ends on similar data.

All tests write under a tmp dir (never the real vf_runs/). Pure stdlib + pytest.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform.casebase_store import (
    CaseBaseStore, fingerprint, fingerprint_distance, DEAD_END_GAIN,
)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def store(tmp_path):
    return CaseBaseStore(str(tmp_path / "_casebase_outcomes.jsonl"),
                         str(tmp_path / "_casebase_negative.jsonl"))


def _balanced_records(n_per_class=300):
    return ([{"x0": i * 0.1, "x1": -i * 0.2, "target": "a"} for i in range(n_per_class)] +
            [{"x0": i * 0.1, "x1": i * 0.3, "target": "b"} for i in range(n_per_class)])


def _imbalanced_records(n_major=900, n_minor=20):
    return ([{"x0": i * 0.1, "target": "a"} for i in range(n_major)] +
            [{"x0": i * 0.1, "target": "b"} for i in range(n_minor)])


# --------------------------------------------------------------------------- (a) fingerprint
def test_fingerprint_records_and_profile_agree():
    """The records-list path and the profile-dict path must produce the SAME key for the same
    dataset, so the loop's _profile and a raw records list map to one memory cell."""
    recs = _balanced_records(300)            # 600 rows, 2 features, 2 balanced classes
    fp_recs = fingerprint(recs, kind="tabular")
    # the loop builds _profile as n_train/n_val (sum == n_rows), n_features, n_classes, kind
    profile = {"kind": "tabular", "task_type": "binary", "metric": "accuracy",
               "n_train": 480, "n_val": 120, "n_features": 2, "n_classes": 2,
               "class_counts": [300, 300]}
    fp_prof = fingerprint(profile)
    assert fp_recs["key"] == fp_prof["key"]
    assert fp_recs["modality"] == "tabular"
    assert fp_recs["n_classes"] == 2
    assert fp_recs["balance_bucket"] == "balanced"


def test_fingerprint_balance_buckets():
    bal = fingerprint(_balanced_records(300), kind="tabular")
    imb = fingerprint(_imbalanced_records(900, 20), kind="tabular")   # IR = 45 -> severe
    assert bal["balance_bucket"] == "balanced"
    assert imb["balance_bucket"] == "severe"
    assert bal["key"] != imb["key"]          # different data -> different cell


def test_fingerprint_regression_target():
    recs = [{"f": i, "target": float(i) * 1.37 + 0.5} for i in range(400)]
    fp = fingerprint(recs, kind="tabular")
    assert fp["n_classes"] == 1              # continuous target -> regression
    assert fp["n_classes_bucket"] == "reg"
    assert fp["balance_bucket"] == "regression"


def test_fingerprint_modality_from_kind_and_heuristic():
    txt = fingerprint({"kind": "text", "n_rows": 5000, "n_features": 20000, "n_classes": 4})
    assert txt["modality"] == "text"
    # no kind, very wide -> heuristic modality (explicitly labeled, never overrides a real kind)
    wide = fingerprint({"n_rows": 5000, "n_features": 20000, "n_classes": 4})
    assert wide["modality"] == "wide_heuristic"


def test_fingerprint_distance_ordering():
    a = fingerprint(_balanced_records(300), kind="tabular")          # tabular, 1e2 rows
    a_same = fingerprint(_balanced_records(300), kind="tabular")
    a_bigger = fingerprint({"kind": "tabular", "n_rows": 50000, "n_features": 2,
                            "n_classes": 2, "class_counts": [25000, 25000]})
    text = fingerprint({"kind": "text", "n_rows": 600, "n_features": 2, "n_classes": 2,
                        "class_counts": [300, 300]})
    assert fingerprint_distance(a, a_same) == 0.0
    # same modality, different size < different modality
    assert fingerprint_distance(a, a_bigger) < fingerprint_distance(a, text)
    assert fingerprint_distance(a, a_bigger) > 0.0


# --------------------------------------------------------------------------- (b) durability
def test_outcome_log_durable_and_corrupt_tolerant(store, tmp_path):
    fp = fingerprint(_balanced_records(300), kind="tabular")
    store.record_outcome(fp, "hist_gbm", gain=0.1, cost=4.0, ts=0)
    store.record_outcome(fp, "logistic", gain=0.02, cost=0.5, ts=1)
    assert store.counts()["outcomes"] == 2
    # a fresh store on the same path sees the persisted rows (durable across "process restart")
    store2 = CaseBaseStore(store.outcome_path, store.negative_path)
    assert store2.counts()["outcomes"] == 2
    # inject a corrupt line; reads must skip it, not crash
    with open(store.outcome_path, "a") as f:
        f.write("{not json}\n")
    assert store2.counts()["outcomes"] == 2          # corrupt line skipped
    store2.record_outcome(fp, "svc_rbf", gain=0.05, cost=2.0, ts=2)
    assert store2.counts()["outcomes"] == 3          # append still works after corruption


# --------------------------------------------------------------------------- (c) warm start
def test_warm_start_cold_is_empty(store):
    fp = fingerprint(_balanced_records(300), kind="tabular")
    cold = store.warm_start(fp)
    assert cold["families"] == {}
    assert cold["order"] == []
    assert cold["n_neighbors"] == 0


def test_warm_start_priors_improve_after_runs(store):
    """The core claim: warm_start returns sensible priors, and they IMPROVE (become informative
    and correctly ranked) after simulated runs on data like this."""
    fp = fingerprint(_balanced_records(300), kind="tabular")
    # before: no signal
    assert store.warm_start(fp)["families"] == {}
    # simulate: hist_gbm consistently gains a lot, logistic a little, knn nothing
    for ts in range(4):
        store.record_outcome(fp, "hist_gbm", gain=0.10 + 0.005 * ts, cost=4.0, ts=ts)
        store.record_outcome(fp, "logistic", gain=0.03, cost=0.5, ts=ts)
        store.record_outcome(fp, "knn", gain=0.0, cost=1.5, ts=ts)
    warm = store.warm_start(fp)
    fams = warm["families"]
    # priors are now populated and reflect the realized gains
    assert set(fams) >= {"hist_gbm", "logistic", "knn"}
    assert fams["hist_gbm"]["prior_gain"] > fams["logistic"]["prior_gain"] > fams["knn"]["prior_gain"]
    assert fams["hist_gbm"]["win_rate"] == 1.0
    assert fams["knn"]["win_rate"] == 0.0
    # highest-gain family is hist_gbm; the dead end ranks last and is on the avoid list
    best_gain = max(fams, key=lambda f: fams[f]["prior_gain"])
    assert best_gain == "hist_gbm"
    assert warm["order"][-1] == "knn"
    assert "knn" in warm["avoid"]


def test_warm_start_prefers_nearest_fingerprint(store):
    """Distance-weighting: an EXACT-match dataset's history should dominate a merely-similar
    dataset's history when forming the prior for the target."""
    target = fingerprint(_balanced_records(300), kind="tabular")            # 600 rows
    similar = fingerprint({"kind": "tabular", "n_rows": 60000, "n_features": 2,
                           "n_classes": 2, "class_counts": [30000, 30000]})  # same modality, far size
    # On the EXACT dataset, family X is great. On the similar (but bigger) dataset, X is a dead end.
    for ts in range(3):
        store.record_outcome(target, "family_x", gain=0.20, cost=2.0, ts=ts)
        store.record_outcome(similar, "family_x", gain=0.0, cost=2.0, ts=ts)
    warm = store.warm_start(target)
    # the exact-match high gain must dominate -> prior_gain well above the similar-dataset zero
    assert warm["families"]["family_x"]["prior_gain"] > 0.10


# --------------------------------------------------------------------------- (d) mine ledger
def test_mine_promotion_ledger_attributes_and_skips(store, tmp_path):
    """Mining folds CERTIFIED rows that carry (family, fp) into priors and SKIPS rows that the
    frozen ledger schema leaves unattributable. It must never invent a family/profile."""
    ledger = tmp_path / "_promotion_ledger.jsonl"
    fp = fingerprint(_balanced_records(300), kind="tabular")
    rows = [
        # attributable win (newer writer added family+fp): observed-theta = 0.05 margin
        {"plan_hash": "sha256:aa", "decision": "certified", "certified": True,
         "metric": "accuracy", "theta": 0.90, "observed": 0.95, "lower_bound": 0.93,
         "family": "hist_gbm", "fp": fp, "ts": 1},
        # attributable non-win -> negative memory for that family on that data
        {"plan_hash": "sha256:bb", "decision": "reject", "certified": False,
         "metric": "accuracy", "theta": 0.90, "observed": 0.80,
         "family": "knn", "fp": fp, "ts": 2},
        # UNATTRIBUTABLE: frozen-schema row with no family / no fp -> must be SKIPPED
        {"plan_hash": "sha256:cc", "decision": "certified", "certified": True,
         "metric": "accuracy", "theta": 0.90, "observed": 0.97, "lower_bound": 0.95, "ts": 3},
    ]
    with open(ledger, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = store.mine_promotion_ledger(str(ledger))
    assert summary["total"] == 3
    assert summary["mined"] == 2          # the win + the non-win (folded as negative)
    assert summary["skipped"] == 1        # the unattributable certified row

    warm = store.warm_start(fp)
    assert "hist_gbm" in warm["families"]
    assert warm["families"]["hist_gbm"]["prior_gain"] == pytest.approx(0.05, abs=1e-6)
    # knn came in only as a non-win -> it is a dead end / on the avoid list
    assert "knn" in warm["avoid"]


def test_mine_with_profile_resolver(store, tmp_path):
    """When the frozen schema lacks (family, fp) but the caller has a side-index, a
    profile_resolver maps plan_hash -> (family, fp); rows it can't resolve are still skipped."""
    ledger = tmp_path / "_promotion_ledger.jsonl"
    fp = fingerprint(_balanced_records(300), kind="tabular")
    rows = [
        {"plan_hash": "sha256:dd", "decision": "certified", "certified": True,
         "theta": 0.8, "observed": 0.9, "ts": 1},
        {"plan_hash": "sha256:ee", "decision": "certified", "certified": True,
         "theta": 0.8, "observed": 0.85, "ts": 2},
    ]
    with open(ledger, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    side_index = {"sha256:dd": ("random_forest", fp)}     # only one is resolvable

    def resolver(row):
        return side_index.get(row.get("plan_hash"))

    summary = store.mine_promotion_ledger(str(ledger), profile_resolver=resolver)
    assert summary["mined"] == 1 and summary["skipped"] == 1
    warm = store.warm_start(fp)
    assert "random_forest" in warm["families"]
    assert warm["families"]["random_forest"]["prior_gain"] == pytest.approx(0.10, abs=1e-6)


def test_mine_real_promotion_ledger_does_not_crash(store):
    """Sanity: the real vf_runs/_promotion_ledger.jsonl (frozen schema, no family) mines to all
    SKIPPED without crashing and without fabricating any prior."""
    real = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "vf_runs", "_promotion_ledger.jsonl")
    if not os.path.exists(real):
        pytest.skip("no real promotion ledger present")
    summary = store.mine_promotion_ledger(real)
    assert summary["total"] >= 1
    # frozen schema carries no family -> every row unattributable -> nothing mined, nothing guessed
    assert summary["mined"] == 0
    assert summary["skipped"] == summary["total"]
    assert store.counts()["outcomes"] == 0


# --------------------------------------------------------------------------- (e) negative memory
def test_negative_memory_records_dead_ends_automatically(store):
    """A zero-gain outcome auto-writes a negative-memory record; warm_start surfaces it as an
    avoid + penalty so the proposer can deprioritize the dead end."""
    fp = fingerprint(_balanced_records(300), kind="tabular")
    store.record_outcome(fp, "deep_model", gain=DEAD_END_GAIN / 2, cost=10.0, ts=0)
    assert store.counts()["negatives"] == 1
    warm = store.warm_start(fp)
    assert "deep_model" in warm["avoid"]
    assert warm["families"]["deep_model"]["penalty"] > 0.0


def test_negative_memory_steers_away_on_similar_data(store):
    """A family that died on a near fingerprint carries a penalty when warm-starting a similar
    (non-identical) dataset, so the dead end is deprioritized even on data we have not seen."""
    seen = fingerprint(_balanced_records(300), kind="tabular")
    # explicit approach-level negative on the seen data
    for ts in range(3):
        store.record_negative(seen, "tfidf+mlp", reason="diverged", ts=ts)
    # a slightly different dataset (same modality + classes, mild imbalance via class_counts)
    near = fingerprint({"kind": "tabular", "n_rows": 600, "n_features": 2, "n_classes": 2,
                        "class_counts": [400, 200]})
    warm = store.warm_start(near)
    assert "tfidf+mlp" in warm["families"]
    assert warm["families"]["tfidf+mlp"]["penalty"] > 0.0
    assert "tfidf+mlp" in warm["avoid"]


def test_record_outcome_validates_inputs(store):
    fp = fingerprint(_balanced_records(10), kind="tabular")
    with pytest.raises(ValueError):
        store.record_outcome({"not": "a fp"}, "fam", 0.1, 1.0)
    with pytest.raises(ValueError):
        store.record_outcome(fp, "", 0.1, 1.0)
