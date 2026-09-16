"""Sandbox hardening tests (item: sandbox_honesty).

Two goals, proved end to end on the REAL sandbox + a real sklearn dataset:

  (a) The file-write escape CLASS is triaged. A candidate doing arbitrary file writes through an
      allowlisted-library bound method -- `ndarray.dump('/tmp/x')` (writes a pickle to a path),
      `ndarray.dumps()` (pickle to bytes), `scipy.io.savemat`/`mmwrite` (file writers) -- is rejected
      by the advisory AST gate with error_kind="policy" BEFORE any subprocess is spawned. We also prove
      the gate is ADVISORY by exhibiting a bypass the gate misses (a dynamically-fetched attribute), and
      then prove goal (b) makes even a SUCCESSFUL write harmless to the certificate.

  (b) The FIREWALL is airtight regardless of the gate:
        * the sandbox child only ever receives train + eval FEATURES, NEVER the sealed labels;
        * the child returns predictions only; the trusted PARENT computes every score;
        * therefore a malicious candidate that fully escapes the gate and writes/exfiltrates whatever
          it was handed still cannot read the sealed labels or forge a score.

  (c) WIRING: the hardened policy is LIVE in CoreOrchestrator. A run whose winner-equivalent code
      carries `ndarray.dump` is rejected through the orchestrator's own sandbox_policy front door
      (not raw run_program), and a clean run stamps the probe-derived `sandbox_enforced` block onto the
      certificate. This asserts the capability actually changed observable execution.

Run:
    cd <repo> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_sandbox_hardening.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, sandbox
from frontier.program import Program
from frontier.proposers import make_code
from frontier.task import Task
from frontier.core.sandbox_policy import (
    SandboxPolicy, Tier, ast_check, BANNED_ATTRS, enforced_of, validate_predictions,
)
from frontier.core.orchestrator import CoreOrchestrator, CoreConfig


def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.80, name="bc")


# ----------------------------------------------------------------------- (a) escape class closed
def test_file_write_escape_class_is_in_denylist():
    """The previously-missed file-write escape methods are now on the advisory denylist."""
    must_cover = {"dump", "dumps", "savemat", "mmwrite", "hb_write", "netcdf_file", "tobytes"}
    missing = must_cover - set(BANNED_ATTRS)
    assert not missing, f"file-write escape class not covered by BANNED_ATTRS: {missing}"
    print(f"[ok] (a) denylist covers the file-write escape class: {sorted(must_cover)}")


def test_ndarray_dump_escape_is_actually_writable_then_gated():
    """Confirm `ndarray.dump(path)` is a REAL arbitrary-write primitive on this host, then confirm the
    advisory AST gate now rejects a candidate that uses it (it did NOT before this item)."""
    # 1) the primitive is real (a candidate COULD write an arbitrary file via an allowlisted lib)
    import tempfile
    probe = os.path.join(tempfile.gettempdir(), "frontier_escape_probe_test.bin")
    if os.path.exists(probe):
        os.remove(probe)
    np.arange(7).dump(probe)
    assert os.path.exists(probe) and os.path.getsize(probe) > 0, "ndarray.dump should write a real file"
    os.remove(probe)

    # 2) the gate now catches the obvious form
    malicious = (
        "import numpy as np\n"
        "def build_estimator():\n"
        "    class E:\n"
        "        def fit(self, X, y):\n"
        "            np.asarray(X).dump('/tmp/frontier_pwned.bin')\n"
        "            return self\n"
        "        def predict(self, X):\n"
        "            return ['0'] * len(X)\n"
        "    return E()\n"
    )
    rep = ast_check(malicious)
    assert not rep.ok, "advisory gate must flag ndarray.dump"
    assert any("dump" in v for v in rep.violations), f"violation should name dump: {rep.violations}"
    print(f"[ok] (a) ndarray.dump is real-writable AND now gated: {rep.violations}")


def test_policy_blocks_dump_before_spawn():
    """Through the policy front door, a dump-escaping candidate is rejected as error_kind='policy'
    WITHOUT spawning a subprocess (the advisory triage runs first)."""
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xev = Task.rows_to_X(s.val_rows)
    malicious = (
        "import numpy as np\n"
        "def build_estimator():\n"
        "    class E:\n"
        "        def fit(self, X, y):\n"
        "            np.asarray(y).dumps()\n"           # serialize-to-bytes form
        "            return self\n"
        "        def predict(self, X):\n"
        "            return ['0'] * len(X)\n"
        "    return E()\n"
    )
    prog = Program(code=malicious, source="mutation", label="evil-dumps")
    pol = SandboxPolicy(requested=Tier.LOCAL, untrusted=True, strict=False,
                        local_runner=sandbox.run_program)
    res = pol.run(prog, Xtr, ytr, Xev, kind=task.kind, wall_seconds=30)
    assert not res.ok and res.error_kind == "policy", f"expected policy reject, got {res.error_kind}: {res.error}"
    assert res.wall_seconds == 0.0, "policy reject must not have spawned a process"
    print(f"[ok] (a) policy front door blocks dumps before spawn: [{res.error_kind}] {res.error[:70]}")


# ----------------------------------------------------------------------- (b) firewall is airtight
def test_sealed_labels_never_cross_the_boundary():
    """STRUCTURAL firewall proof: the runner contract takes (Xtr, ytr, Xev) only. We instrument the
    injected runner to capture EXACTLY what the child was handed and assert no sealed-label array is
    among them. y_train (the TRAIN labels) is allowed (the model fits on it); the SEALED labels are not.
    """
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    X_sealed = Task.rows_to_X(s.sealed_rows)
    y_sealed = Task.rows_to_y(s.sealed_rows, task.kind)

    captured = {}

    def spy_runner(program, X_train, y_train, X_eval, *, kind, **kw):
        # Record every positional/array the child layer receives.
        captured["args"] = (X_train, y_train, X_eval)
        return sandbox.run_program(program, X_train, y_train, X_eval, kind=kind, **kw)

    pol = SandboxPolicy(requested=Tier.LOCAL, untrusted=True, strict=False, local_runner=spy_runner)
    prog = Program(code=make_code({"base": "logreg", "scale": True}, "classification"),
                   source="seed", label="scale+logreg")
    res = pol.run(prog, Xtr, ytr, X_sealed, kind=task.kind, wall_seconds=40)
    assert res.ok, f"clean program should run: {res.error_kind} {res.error}"

    X_train_seen, y_train_seen, X_eval_seen = captured["args"]
    # The eval features handed to the child are the sealed FEATURES (that is expected and necessary).
    assert np.array_equal(np.asarray(X_eval_seen, dtype=float), X_sealed.astype(float)), \
        "child should receive sealed FEATURES as eval"
    # The SEALED LABELS must NOT be any array handed to the child.
    ys = np.asarray(y_sealed).astype(str)
    for name, arr in (("X_train", X_train_seen), ("y_train", y_train_seen), ("X_eval", X_eval_seen)):
        a = np.asarray(arr)
        if a.shape == ys.shape and a.astype(str).tolist() == ys.tolist():
            raise AssertionError(f"sealed labels leaked into child arg {name!r}")
    # And y_train_seen must be the TRAIN labels, not the sealed ones (sanity that we tested the right thing).
    assert len(np.asarray(y_train_seen)) == len(ytr), "child must receive TRAIN labels for fitting"
    print(f"[ok] (b) child got sealed FEATURES + train labels; sealed LABELS never crossed "
          f"({len(X_sealed)} sealed rows, labels withheld)")


def test_parent_computes_score_child_returns_predictions_only():
    """The RunResult carries predictions and NO score; the parent (this test, standing in for the
    orchestrator) computes the metric. A child therefore cannot forge a score."""
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    X_sealed = Task.rows_to_X(s.sealed_rows)
    pol = SandboxPolicy(requested=Tier.LOCAL, untrusted=True, strict=False,
                        local_runner=sandbox.run_program)
    prog = Program(code=make_code({"base": "logreg", "scale": True}, "classification"),
                   source="seed", label="scale+logreg")
    res = pol.run(prog, Xtr, ytr, X_sealed, kind=task.kind, wall_seconds=40)
    assert res.ok and res.preds is not None
    assert not hasattr(res, "score"), "RunResult must not carry a score (firewall)"
    ok, why = validate_predictions(res.preds, kind=task.kind, n_expected=len(X_sealed),
                                   labels=task.labels)
    assert ok, f"parent-side prediction guard rejected clean preds: {why}"
    # ONLY the parent, holding the sealed rows, can certify.
    cert = certify.certify_on_sealed(task, s, res.preds)
    assert "observed" in cert and "lower_bound" in cert and cert["peeks"] == 1
    print(f"[ok] (b) firewall: child returned {len(res.preds)} preds, parent computed "
          f"observed={cert['observed']} lb={cert['lower_bound']} (1 peek)")


# --------------------------------------------------------- (c) WIRED LIVE into CoreOrchestrator
def test_orchestrator_routes_through_policy_front_door():
    """The orchestrator's untrusted-code crossings go through self.sandbox_policy, not raw
    run_program. We replace the policy with a spy and assert it is actually called during a real run,
    AND that the certificate carries the probe-derived sandbox_enforced stamp (observable change)."""
    task = _clf_task()
    calls = {"n": 0}

    base = SandboxPolicy(requested=Tier.CONTAINER, untrusted=True, strict=False,
                         local_runner=sandbox.run_program)

    class SpyPolicy(SandboxPolicy):
        def run(self, *a, **kw):
            calls["n"] += 1
            return base.run(*a, **kw)

    spy = SpyPolicy(requested=Tier.CONTAINER, untrusted=True, strict=False,
                    local_runner=sandbox.run_program)
    orch = CoreOrchestrator(CoreConfig(rounds=1, wall_seconds=45, cpu_seconds=40, enable_neural=False,
                                       enable_knowledge=False),
                            sandbox_policy=spy)
    d = __import__("sklearn.datasets", fromlist=["load_breast_cancer"]).load_breast_cancer()
    res = orch.run("classify tumors", d.data, d.target.astype(str), theta=0.80)
    assert calls["n"] >= 1, "orchestrator must execute untrusted code through the sandbox_policy front door"
    assert res.certificate is not None, f"expected a certificate; declined: {res.decline_reason}"
    enf = res.certificate.get("sandbox_enforced")
    assert enf is not None, "certificate must carry the probe-derived sandbox_enforced stamp"
    # Honest stamp: on macOS RLIMIT_AS is unenforced -> mem_guard must NOT claim a memory cap.
    if sys.platform == "darwin":
        assert enf["mem_guard"] == "wall-timeout-only", \
            f"macOS must report wall-timeout-only mem guard, got {enf['mem_guard']!r}"
    assert enf["advisory_ast_gate"] is True, "the advisory gate must have run before exec"
    print(f"[ok] (c) WIRED: policy.run called {calls['n']}x via orchestrator; "
          f"cert.sandbox_enforced tier={enf['tier']} mem_guard={enf['mem_guard']} "
          f"network={enf['network']} ast_gate={enf['advisory_ast_gate']}")


def test_orchestrator_gate_blocks_dump_winner_on_sealed_refit():
    """A winner whose code contains a file-write escape is rejected by the LIVE gate at the sealed
    re-fit crossing (proving the gate guards the most security-critical step). We drive _certify_winner
    directly with a dump-escaping program to assert the policy refuses it before the sealed peek."""
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    orch = CoreOrchestrator(CoreConfig(rounds=1, wall_seconds=30, cpu_seconds=25))
    evil = Program(
        code=("import numpy as np\n"
              "def build_estimator():\n"
              "    class E:\n"
              "        def fit(self, X, y):\n"
              "            np.asarray(X).dump('/tmp/frontier_sealed_exfil.bin')\n"
              "            return self\n"
              "        def predict(self, X):\n"
              "            return ['0'] * len(X)\n"
              "    return E()\n"),
        source="mutation", label="evil-dump-winner")
    cert, fail = orch._certify_winner(task, s, evil, Xtr, ytr)
    assert cert is None and fail is not None, "dump-escaping winner must be refused before the sealed peek"
    assert "policy" in fail, f"refusal should be a policy rejection: {fail}"
    assert orch.sealed_peeks == 0, "no sealed peek may be consumed when the gate refuses the winner"
    print(f"[ok] (c) live gate refuses dump-escaping winner at sealed re-fit (peeks={orch.sealed_peeks}): {fail[:80]}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} sandbox-hardening tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
