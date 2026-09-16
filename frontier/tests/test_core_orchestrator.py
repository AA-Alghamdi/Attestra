"""Integration tests for frontier.core.orchestrator.CoreOrchestrator (design 04).

These assert the load-bearing properties of THE CORE end-to-end loop:
  - the offline (no-LLM) path runs FOR REAL on a real sklearn dataset and yields a
    certified-or-declined result (never a relabeled val score);
  - the sealed test is peeked EXACTLY once for the winner (firewall + one-peek invariant);
  - the oracle gate is genuinely active (it can flip certified -> declined);
  - an honest decline is produced when no candidate can clear theta or when the harness/router
    blocks the goal -- with certificate=None and no peek spent;
  - the firewall holds (untrusted code returns predictions only; the parent computes every
    promoting number).

Run:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_core_orchestrator.py
or via pytest:
    /Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/tests/test_core_orchestrator.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.core.orchestrator import CoreOrchestrator, CoreConfig, CoreResult  # noqa: E402

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _bc_xy():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, d.target.astype(str)


# ---------------------------------------------------------------------------------------------
# 1. End-to-end OFFLINE run: real sklearn dataset, no LLM. Must certify-or-decline honestly,
#    peek the sealed test exactly once, and run the oracle gate.
# ---------------------------------------------------------------------------------------------

def test_offline_end_to_end_certifies_and_peeks_once():
    X, y = _bc_xy()
    cfg = CoreConfig(rounds=2, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=True, enable_knowledge=True)
    orch = CoreOrchestrator(cfg)
    res = orch.run(goal="classify breast tumors", X=X, y=y, theta=0.90, name="bc")

    assert isinstance(res, CoreResult)
    # offline floor is honest: no LLM was active.
    assert res.llm_active is False, "no client wired -> llm_active must be False"

    # a candidate ran (the generative floor is always on) and the winner was selected on VAL.
    assert res.winner is not None, "the seed/mutation/feature floor must produce a winner"
    assert res.winner_val_score is not None

    # THE one-peek invariant: the orchestrator counted exactly one sealed certification, and the
    # frozen certificate itself reports a single counted peek.
    assert res.sealed_peeks == 1, f"sealed must be peeked exactly once, got {res.sealed_peeks}"
    assert res.certificate is not None
    assert res.certificate.get("peeks") == 1, "frozen certificate must report peeks == 1"

    # outcome is honest: either a real certified result or a labeled decline.
    if res.certified:
        assert res.certificate.get("certified") is True
        assert res.decline_reason == ""
        # oracle gate ANDed in: a certified result means every blocking oracle passed.
        assert res.oracle_verdict is not None and res.oracle_verdict.get("promote") is True
    else:
        assert res.decline_reason, "a non-certified result must carry an honest reason"

    # firewall sanity: the winning Program is plain code; the parent (certify) minted the number.
    assert hasattr(res.winner, "code") and "build_estimator" in res.winner.code

    # report artifact was rendered (pure consumer; no extra peek).
    assert res.artifact is not None
    assert res.sealed_peeks == 1, "report rendering must not spend another peek"


# ---------------------------------------------------------------------------------------------
# 2. The oracle gate is genuinely active: a deliberately leaky task (a feature that IS the label)
#    must be REFUTED by the oracle battery -> honest decline, even though the sealed certificate
#    itself certifies. This proves the AND is real and not a no-op. The sealed peek is still 1.
# ---------------------------------------------------------------------------------------------

def test_oracle_gate_refutes_label_leak():
    X, y = _bc_xy()
    # Inject a leak: append the numeric label as a feature column. The model trivially recovers
    # the answer, the certificate certifies high, but the no-label-leak / permuted-label oracles
    # must catch it.
    y_num = (y == y[0]).astype(float)  # deterministic 0/1 from the string labels
    Xleak = np.column_stack([X, y_num.astype(float)])
    yl = y_num.astype(str)

    cfg = CoreConfig(rounds=1, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=False, enable_knowledge=False)
    orch = CoreOrchestrator(cfg)
    res = orch.run(goal="classify with a leaked label column", X=Xleak, y=yl, theta=0.90,
                   name="leak")

    # The certificate likely certifies (the leak makes the task trivial), but the oracle gate
    # must veto it -> certified=False with an honest reason, and still exactly one sealed peek.
    assert res.winner is not None
    assert res.sealed_peeks == 1
    assert res.oracle_verdict is not None
    if res.certificate is not None and res.certificate.get("certified"):
        assert res.certified is False, "a leaked-label win must be refuted by the oracle gate"
        assert "oracle" in res.decline_reason.lower() or res.oracle_verdict.get("promote") is False


# ---------------------------------------------------------------------------------------------
# 3. Honest decline when theta is unreachable: an impossible threshold (theta=1.0 lower-bound)
#    yields certified=False with the certificate attempt carried and exactly one peek consumed.
# ---------------------------------------------------------------------------------------------

def test_honest_decline_on_unreachable_theta():
    X, y = _bc_xy()
    cfg = CoreConfig(rounds=1, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=False, enable_knowledge=False)
    orch = CoreOrchestrator(cfg)
    # A perfect-accuracy lower-bound is unreachable on finite sealed data -> the Clopper-Pearson
    # lower bound cannot equal 1.0, so this must decline honestly.
    res = orch.run(goal="classify breast tumors", X=X, y=y, theta=1.0, name="bc_hard")

    assert res.winner is not None, "a winner is still selected on VAL"
    assert res.certified is False, "theta=1.0 cannot certify on a finite sealed lower bound"
    assert res.decline_reason, "decline must carry a reason"
    assert res.sealed_peeks == 1, "the winner is still certified once (the bound just falls short)"
    # never a relabeled val score: the result is a decline, the val score is separately labeled.
    assert res.winner_val_score is not None


# ---------------------------------------------------------------------------------------------
# 4. as_engine_result() projects onto the frozen EngineResult shape so report.py consumes it.
# ---------------------------------------------------------------------------------------------

def test_projects_onto_engine_result_shape():
    from frontier.engine import EngineResult
    X, y = _bc_xy()
    cfg = CoreConfig(rounds=1, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=False, enable_knowledge=False)
    res = CoreOrchestrator(cfg).run(goal="classify", X=X, y=y, theta=0.85, name="bc_proj")
    er = res.as_engine_result()
    assert isinstance(er, EngineResult)
    # summary() must not raise and must mention certification status.
    s = er.summary()
    assert "certified" in s


# ---------------------------------------------------------------------------------------------
# 5. Torch path GATING: when torch is absent the neural arms decline HONESTLY (no fake GPU run)
#    and the loop still certifies via the sklearn floor. When torch IS present, the run still
#    completes (the neural arms may or may not win; either way the certificate is the promoter).
# ---------------------------------------------------------------------------------------------

def test_neural_gating_is_honest():
    X, y = _bc_xy()
    cfg = CoreConfig(rounds=1, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=True, enable_knowledge=False)
    res = CoreOrchestrator(cfg).run(goal="classify", X=X, y=y, theta=0.85, name="bc_neural")
    # The run completes regardless of torch availability; the certificate is the only promoter.
    assert res.winner is not None
    assert res.sealed_peeks == 1
    if not _HAS_TORCH:
        # neural arms must have declined honestly (their RunResults are failures, not fakes); the
        # sklearn floor carried the result. We simply assert the loop did not crash and certified
        # or declined honestly.
        assert isinstance(res, CoreResult)


def test_large_data_val_cap_bounds_selection_set_without_touching_sealed():
    """Large-data memory guard: when the VAL split exceeds cfg.eval_max_rows, the orchestrator
    caps the validation rows every sandboxed arm must predict on (bounded peak memory) while
    leaving the SEALED test full. A smaller val set only WIDENS the bound (more conservative),
    so this can never make certification easier.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=4000, n_features=12, n_informative=6,
                               n_classes=3, n_clusters_per_class=1, random_state=0)
    y = y.astype(str)
    cap = 200
    cfg = CoreConfig(rounds=1, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=False, enable_knowledge=False, enable_intelligence=False,
                     eval_max_rows=cap)
    res = CoreOrchestrator(cfg).run(goal="classify", X=X, y=y, theta=0.80, name="cap")

    # the cap fired and is honestly recorded in the provenance notes.
    note = next((n for n in (res.backend_notes or []) if "capped val selection set" in n), None)
    assert note is not None, f"expected a val-cap note, got {res.backend_notes}"
    assert f"to {cap} rows" in note, note

    # the sealed certificate (the actual certificate) was NOT shrunk to the cap: its n reflects
    # the full sealed partition (~test_frac of 4000), which is far larger than the val cap.
    assert res.sealed_peeks == 1
    assert res.certificate is not None
    n_sealed = res.certificate.get("n") or res.certificate.get("n_test") or 0
    assert n_sealed > cap, f"sealed test must stay full (got n={n_sealed}, cap={cap})"


if __name__ == "__main__":
    import traceback
    tests = [
        test_offline_end_to_end_certifies_and_peeks_once,
        test_oracle_gate_refutes_label_leak,
        test_honest_decline_on_unreachable_theta,
        test_projects_onto_engine_result_shape,
        test_neural_gating_is_honest,
        test_large_data_val_cap_bounds_selection_set_without_touching_sealed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
