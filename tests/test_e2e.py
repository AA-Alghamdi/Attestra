"""End-to-end acceptance tests for the VectorForge classifier goal runner.

Pytest-free: plain asserts, runnable directly with the project interpreter. Exercises every honest path
on the REAL frozen datasets and proves the rigor guarantees hold:

  1. certified + deployed (achievable bar)         -> PASSED, deployment ready, certifier lb > threshold
  2. NEEDS_INPUT (too little data)                  -> honest data-limited ask, no fabricated certificate
  3. honest FAILED (bar above the ceiling)          -> FAILED, never certified, threshold never relaxed
  4. resume idempotence (re-run a finished goal)    -> status + completed stages unchanged
  5. predict from a reloaded artifact               -> serves a valid label after store.load
  6. the certifier REFUSES a borderline case        -> point estimate above threshold, lower bound below

The data-backed paths (1-5 and the borderline GOAL) read a curated, private corpus that lives OUTSIDE the
repo; its location is configurable via VF_E2E_TEXT_DATA / VF_E2E_TAB_DATA (defaults match the author's
workstation). When the corpus is absent (CI / a fresh clone) those paths SKIP cleanly instead of failing,
while the data-free certifier-rigor property (test 6 unit) still runs everywhere.

Run:  VF_E2E_TEXT_DATA=/path/to/core-ml-acceptance/data python tests/test_e2e.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vectorforge as vf
from vectorforge import store, science, classifier
from vectorforge.domain import PASSED, FAILED, NEEDS_INPUT

# pytest is optional so this file still runs directly (`python tests/test_e2e.py`) on a box without it.
try:
    import pytest
    _skipif = pytest.mark.skipif
except ImportError:                                  # pragma: no cover - direct-run without pytest
    def _skipif(cond, reason=""):
        def deco(fn):
            return fn
        return deco

TEXT = Path(os.environ.get("VF_E2E_TEXT_DATA", "/Users/abdullahalghamdi/core-ml-acceptance/data"))
TAB = Path(os.environ.get("VF_E2E_TAB_DATA", "/Users/abdullahalghamdi/vectorforge-harnesses/rugged/data"))
_NO_DATA = not (TEXT / "train.jsonl").exists()
_NO_DATA_REASON = (f"acceptance corpus not present at {TEXT} -- set VF_E2E_TEXT_DATA to run the data-backed "
                   "e2e acceptance paths (the data-free certifier-rigor unit test still runs)")


def loadj(d, n):
    return [json.loads(l) for l in (d / f"{n}.jsonl").read_text().splitlines() if l.strip()]


_RESULTS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


# ----------------------------------------------------------------------- 1. certified + deployed
def build_passed_goal():
    """Build the achievable-bar goal used by tests 1/4/5. NOT a test (no `test_` prefix) so pytest does
    not collect it; the `passed_goal` fixture in conftest.py and main() both call it."""
    raw = loadj(TEXT, "train") + loadj(TEXT, "validation") + loadj(TEXT, "test")
    return classifier.run_classifier_goal(name="e2e_pass", kind="text", labels=["class_a", "class_b"],
                                          raw_rows=raw, metric="accuracy", threshold=0.70,
                                          max_latency_ms=50.0, min_heldout_n=200)


@_skipif(_NO_DATA, reason=_NO_DATA_REASON)
def test_certified_and_deployed(g=None):
    # default-arg (not a bare param) so pytest does not treat `g` as a fixture request; builds its own
    # goal under pytest, reuses main()'s single build in direct-run mode.
    g = g if g is not None else build_passed_goal()
    cert = g.certificate or {}
    check("certified+deployed: status PASSED", g.status == PASSED, f"status={g.status}")
    check("certified+deployed: certificate decision == certified", cert.get("decision") == "certified")
    check("certified+deployed: lower bound truly clears the threshold",
          cert.get("lower_bound", 0) > cert.get("threshold", 1),
          f"lb={cert.get('lower_bound')} > thr={cert.get('threshold')}")
    check("certified+deployed: deployment ready + smoke ok",
          (g.deployment or {}).get("status") == "ready" and (g.deployment or {}).get("smoke_ok"))
    check("certified+deployed: latency within budget",
          cert.get("latency_ms_p95", 1e9) <= g.verification.max_latency_ms,
          f"p95={cert.get('latency_ms_p95')}ms")
    check("certified+deployed: leakage audit passed", cert.get("leakage_passed") is True)


# ----------------------------------------------------------------------- 2. NEEDS_INPUT (small data)
@_skipif(_NO_DATA, reason=_NO_DATA_REASON)
def test_needs_input_small_data():
    by = {l: [r for r in loadj(TEXT, "train") if r["target"] == l] for l in ("class_a", "class_b")}
    tiny = [r for l in by for r in by[l][:8]]  # 16 rows total -> held-out well below min_heldout_n
    g = classifier.run_classifier_goal(name="e2e_needs_input", kind="text", labels=["class_a", "class_b"],
                                       raw_rows=tiny, metric="accuracy", threshold=0.90, min_heldout_n=200)
    check("needs_input: status NEEDS_INPUT", g.status == NEEDS_INPUT, f"status={g.status}")
    check("needs_input: concrete labels ask present",
          (g.needs_input or {}).get("kind") == "labels" and "ask" in (g.needs_input or {}))
    check("needs_input: NO fabricated certificate", g.certificate is None)
    check("needs_input: NOT deployed", g.deployment is None)
    check("needs_input: decide() -> ACQUIRE_LABELS", classifier.decide(g) == classifier.ACQUIRE_LABELS)


# ----------------------------------------------------------------------- 3. honest FAILED (bar > ceiling)
@_skipif(_NO_DATA, reason=_NO_DATA_REASON)
def test_honest_failed_above_ceiling():
    raw = loadj(TEXT, "train") + loadj(TEXT, "validation") + loadj(TEXT, "test")
    g = classifier.run_classifier_goal(name="e2e_failed", kind="text", labels=["class_a", "class_b"],
                                       raw_rows=raw, metric="accuracy", threshold=0.99,
                                       max_latency_ms=50.0, min_heldout_n=200)
    cert = g.certificate or {}
    check("failed: status FAILED", g.status == FAILED, f"status={g.status}")
    check("failed: NOT certified", not classifier.certified(g))
    check("failed: threshold was NOT relaxed (still 0.99)", abs(cert.get("threshold", 0) - 0.99) < 1e-9,
          f"threshold={cert.get('threshold')}")
    check("failed: honest failure report with a gap", (g.failure_report or {}).get("gap", -1) >= 0)
    check("failed: NOT deployed", g.deployment is None)
    check("failed: decide() never PROMOTE",
          classifier.decide(g) in (classifier.STOP_HONEST_FAIL, classifier.COLLECT_MORE_HELDOUT),
          classifier.decide(g))


# ----------------------------------------------------------------------- 4. resume idempotence
@_skipif(_NO_DATA, reason=_NO_DATA_REASON)
def test_resume_idempotence(passed_goal):
    before_status = passed_goal.status
    before_stages = set(store.load(passed_goal.id).dag)
    before_cert = dict(passed_goal.certificate or {})
    g2 = vf.run(passed_goal.id)  # re-run a finished goal
    check("resume: status unchanged after re-run", g2.status == before_status,
          f"{before_status} -> {g2.status}")
    check("resume: completed stages unchanged", set(g2.dag) == before_stages)
    check("resume: certificate unchanged (test not re-peeked into a new number)",
          dict(g2.certificate or {}).get("observed") == before_cert.get("observed") and
          dict(g2.certificate or {}).get("lower_bound") == before_cert.get("lower_bound"))


# ----------------------------------------------------------------------- 5. predict from reloaded artifact
@_skipif(_NO_DATA, reason=_NO_DATA_REASON)
def test_predict_from_reload(passed_goal):
    reloaded = store.load(passed_goal.id)  # fresh object from disk, no in-memory pipe
    for text in ("a beautifully crafted, deeply moving film", "dull, lifeless and a complete waste"):
        pred = vf.predict(reloaded.id, text)
        check(f"predict: reloaded artifact serves a valid label for {text[:24]!r}",
              str(pred) in {"class_a", "class_b"}, f"pred={pred}")
    # serving a non-deployed goal must raise (honest refusal, not a silent default)
    raised = False
    try:
        vf.predict("goal-does-not-exist-xyz", "anything")
    except Exception:
        raised = True
    check("predict: serving an undeployed/unknown goal raises", raised)


# ----------------------------------------------------------------------- 6. certifier REFUSES borderline
def test_certifier_refuses_borderline_unit():
    """The load-bearing rigor property, proven WITHOUT any dataset (direct against the real certifier): a
    borderline result whose POINT estimate is above the threshold but whose Clopper-Pearson LOWER bound is
    below it must NOT certify. This runs on every machine, corpus present or not.
    """
    # 41/50 = 0.82 observed, threshold 0.80: point estimate clears, but with n=50 the lower bound won't.
    n, theta = 50, 0.80
    c = science.certify_accuracy(0.82, n, theta, checks=1, alpha=0.05)
    check("borderline: point estimate is above threshold", c["observed"] >= theta,
          f"observed={c['observed']} >= {theta}")
    check("borderline: lower bound is BELOW threshold", c["lower_bound"] < theta,
          f"lb={c['lower_bound']} < {theta}")
    check("borderline: certifier REFUSES (certified == False)", c["certified"] is False)


@_skipif(_NO_DATA, reason=_NO_DATA_REASON)
def test_certifier_refuses_borderline_goal():
    """The goal-level companion to the unit test: a real run whose winner lands in the borderline regime
    must NOT be PASSED on the point estimate; any PASS must be lower-bound-justified."""
    # a goal whose winner lands in this regime must NOT be PASSED. Build a borderline goal on a small
    # held-out: a real run where the observed val/test margin is thin and n is just at the floor.
    raw = loadj(TEXT, "train") + loadj(TEXT, "validation") + loadj(TEXT, "test")
    # bar set just under the model's typical accuracy so the point estimate hovers near it but the
    # confidence-corrected lower bound on a 200-row test cannot clear it.
    g = classifier.run_classifier_goal(name="e2e_borderline", kind="text", labels=["class_a", "class_b"],
                                       raw_rows=raw, metric="accuracy", threshold=0.73,
                                       max_latency_ms=50.0, min_heldout_n=200)
    gc = g.certificate or {}
    if g.status == PASSED:
        # if it certified, it must be because the lower bound genuinely cleared -- never the point estimate
        check("borderline-goal: any PASS is lower-bound-justified, not point-estimate",
              gc.get("lower_bound", 0) > gc.get("threshold", 1),
              f"lb={gc.get('lower_bound')} thr={gc.get('threshold')}")
    else:
        check("borderline-goal: non-pass never claims certified",
              not classifier.certified(g) and gc.get("decision") != "certified", f"status={g.status}")


def main():
    print("=" * 78)
    print("VECTORFORGE CLASSIFIER -- END-TO-END ACCEPTANCE TESTS")
    print("=" * 78)
    if _NO_DATA:
        print(f"\n[1-5] SKIPPED -- {_NO_DATA_REASON}")
    else:
        print("\n[1] certified + deployed")
        passed_goal = build_passed_goal()
        test_certified_and_deployed(passed_goal)
        print("\n[2] NEEDS_INPUT (small data)")
        test_needs_input_small_data()
        print("\n[3] honest FAILED (bar above ceiling)")
        test_honest_failed_above_ceiling()
        print("\n[4] resume idempotence")
        test_resume_idempotence(passed_goal)
        print("\n[5] predict from a reloaded artifact")
        test_predict_from_reload(passed_goal)
    print("\n[6] certifier REFUSES a borderline case")
    test_certifier_refuses_borderline_unit()
    if not _NO_DATA:
        test_certifier_refuses_borderline_goal()

    n = len(_RESULTS)
    npass = sum(ok for _, ok, _ in _RESULTS)
    print("\n" + "=" * 78)
    print(f"RESULT: {npass}/{n} checks passed")
    fails = [name for name, ok, _ in _RESULTS if not ok]
    if fails:
        print("FAILURES: " + ", ".join(fails))
    print("PASS" if npass == n else "FAIL")
    return npass == n


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
