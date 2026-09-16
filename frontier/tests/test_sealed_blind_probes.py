"""Phase 6 — sealed-blind adversarial probes.

These are *attack* tests: each one tries to break the one-peek / no-leakage
certification discipline from the outside, using only the public certify/sealed
surface, and asserts the frozen core refuses. They complement the end-to-end
oracle tests in test_core_orchestrator.py (label-leak refutation, unreachable
theta, val-cap) by pinning the lower-level invariants the orchestrator relies on:

  1. the 3-way split partitions are pairwise DISJOINT (no row reused across
     train / val / sealed) -> a memorizing model cannot meet itself on sealed;
  2. a SECOND peek of the same sealed test raises PeekViolation (the one-peek
     counter is real, not a no-op);
  3. certify_on_sealed refuses a prediction vector whose length does not match
     the sealed set (no smuggling a different-sized / re-aligned answer);
  4. the sealed test is content-addressed: tampering with even one sealed row
     changes the digest, so an easier sealed set cannot be swapped in silently.

Run:
    python -m pytest frontier/tests/test_sealed_blind_probes.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify  # noqa: E402
from frontier.task import Task  # noqa: E402
from vfplatform import sealed  # noqa: E402


def _unique_task(n: int = 240, seed: int = 0) -> Task:
    """A classification Task whose every feature vector is unique, so partition
    membership can be checked by exact feature signature."""
    rng = np.random.default_rng(seed)
    # distinct integer rows guarantee unique signatures (no accidental dups that
    # the leakage-dedup would legitimately drop).
    X = (rng.permutation(n * 6).reshape(n, 6)).astype(float)
    y = (np.arange(n) % 3).astype(str)
    return Task(X=X, y=y, kind="classification", theta=0.5, name="probe")


def _sig(row) -> tuple:
    return tuple(row["_x"])


def test_split_partitions_are_pairwise_disjoint():
    """No row may appear in more than one of train / val / sealed."""
    task = _unique_task()
    sp = certify.make_splits(task, seed=0, test_frac=0.30, val_frac=0.20)

    tr = {_sig(r) for r in sp.train_rows}
    va = {_sig(r) for r in sp.val_rows}
    se = {_sig(r) for r in sp.sealed_rows}

    assert tr & va == set(), "train and val overlap -> selection leakage"
    assert tr & se == set(), "train and sealed overlap -> certification leakage"
    assert va & se == set(), "val and sealed overlap -> certification leakage"

    # every sealed row is genuinely held out from everything the search can see.
    seen_by_search = tr | va
    assert se - seen_by_search == se, "a sealed row was visible to the search"

    # the partitions cover the whole dataset (no rows silently vanish); the only
    # legitimate shortfall is audited leakage-dedup, which is zero on unique rows.
    dropped = int(sp.meta.get("leakage_dropped", 0))
    assert dropped == 0
    assert len(tr) + len(va) + len(se) == len(task.X)


def test_second_sealed_peek_raises_peek_violation():
    """The one-peek ledger is real: a second peek of the SAME sealed test is refused."""
    task = _unique_task()
    sp = certify.make_splits(task, seed=0)
    st = sp.sealed_test  # a single SealedTest instance (max_peeks=1)

    preds = [str(r["target"]) for r in st.rows]  # perfect predictions

    def predict_fn(_rows):
        return preds

    # first peek: allowed and counted.
    cert1 = sealed.certify_on_sealed(st, predict_fn, task.theta,
                                     metric=task.metric, labels=list(task.labels),
                                     alpha=0.05, who="probe")
    assert cert1 is not None

    # second peek of the same instance: must raise, not silently re-evaluate.
    with pytest.raises(sealed.PeekViolation):
        sealed.certify_on_sealed(st, predict_fn, task.theta,
                                 metric=task.metric, labels=list(task.labels),
                                 alpha=0.05, who="probe")


def test_certify_rejects_pred_count_mismatch():
    """A prediction vector that does not match the sealed size is refused (no smuggling)."""
    task = _unique_task()
    sp = certify.make_splits(task, seed=0)
    n_sealed = len(sp.sealed_rows)

    too_few = ["0"] * (n_sealed - 1)
    with pytest.raises(ValueError):
        certify.certify_on_sealed(task, sp, too_few)


def test_sealed_test_is_content_addressed_against_tampering():
    """Swapping in an easier sealed row changes the digest -> tampering is evident."""
    task = _unique_task()
    sp = certify.make_splits(task, seed=0)

    st_orig = sealed.SealedTest(sp.sealed_rows, target_key="target", max_peeks=1)

    # flip the label on a single sealed row (an "easier"/altered sealed set).
    tampered = [dict(r) for r in sp.sealed_rows]
    tampered[0] = dict(tampered[0])
    tampered[0]["target"] = "999"
    st_tampered = sealed.SealedTest(tampered, target_key="target", max_peeks=1)

    assert st_orig.digest != st_tampered.digest, (
        "sealed digest must change when sealed content changes (tamper-evidence)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
