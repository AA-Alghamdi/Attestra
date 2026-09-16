"""Tests for frontier.core.execution — the backend-agnostic execution substrate (design 02).

Run standalone (this is what the build gate uses):
    cd <repo> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_core_execution.py

What is exercised FOR REAL on a real dataset (breast cancer, sklearn path, no torch):
  - SklearnBackend via Executor.run produces predictions matching frontier.sandbox.run_program
    bit-for-bit (byte-compatible Phase-0 path), and those predictions certify on the sealed test
    through the frozen certify.py -> vectorforge.science / vfplatform.sealed path (one peek).
  - The back-compat run_program() default path equals frontier.sandbox.run_program.
  - The torch probe degrades HONESTLY: TORCH_SPEC is not runnable here; a torch-tagged Program
    returns error_kind="backend_unavailable" and is NEVER rerouted to sklearn.
  - RemotePodExecutor round-trips the IDENTICAL bundle/runner/CLI through a fake provider (no pod),
    proving GPU is a substrate swap; an infra failure is typed error_kind="remote".
  - The typed error taxonomy (build) and cost model behave as designed.
The torch-needing path (actual training) is skipped cleanly when torch is absent.
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
from frontier.core import execution as ex
from frontier.core.execution import (
    SKLEARN_SPEC, TORCH_SPEC, Job, ResourceLimits, FitSpec,
    LocalSubprocessExecutor, RemotePodExecutor, SklearnBackend, TorchBackend,
    probe_backend, cost_model, serialize_job_to_dir, backend_for,
)


def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")


def _splits_and_arrays(task):
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows)
    ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xva = Task.rows_to_X(s.val_rows)
    Xse = Task.rows_to_X(s.sealed_rows)
    return s, Xtr, ytr, Xva, Xse


def test_sklearn_executor_matches_phase0_sandbox_bitforbit():
    """SklearnBackend via Executor must reproduce frontier.sandbox.run_program predictions exactly."""
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    code = make_code({"base": "logreg", "scale": True}, "classification")
    prog = Program(code=code, source="seed", label="scale+logreg")

    ref = sandbox.run_program(prog, Xtr, ytr, Xva, kind="classification", wall_seconds=40)
    assert ref.ok, f"reference run failed: {ref.error_kind} {ref.error}"

    job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
              backend=SKLEARN_SPEC, limits=ResourceLimits(wall_seconds=40, cpu_seconds=35))
    got = LocalSubprocessExecutor().run(job)
    assert got.ok, f"executor run failed: {got.error_kind} {got.error}"
    assert len(got.preds) == len(ref.preds)
    assert [str(a) for a in got.preds] == [str(b) for b in ref.preds], \
        "sklearn-via-Executor predictions must match Phase-0 sandbox bit-for-bit"
    print(f"[ok] sklearn Executor matches Phase-0 sandbox: {len(got.preds)} identical preds")


def test_backcompat_run_program_equals_sandbox():
    """The default run_program() path equals frontier.sandbox.run_program (back-compat)."""
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    code = make_code({"base": "hist_gbm"}, "classification")
    prog = Program(code=code, source="seed", label="hist_gbm")
    ref = sandbox.run_program(prog, Xtr, ytr, Xva, kind="classification", wall_seconds=50)
    got = ex.run_program(prog, Xtr, ytr, Xva, kind="classification", wall_seconds=50)
    assert ref.ok and got.ok, f"{ref.error_kind}/{got.error_kind}"
    assert [str(a) for a in got.preds] == [str(b) for b in ref.preds]
    print(f"[ok] back-compat run_program == sandbox.run_program ({len(got.preds)} preds)")


def test_sklearn_executor_certifies_on_sealed():
    """End-to-end firewall: executor predictions on the sealed split certify via certify.py."""
    task = _clf_task()
    s, Xtr, ytr, _, Xse = _splits_and_arrays(task)
    code = make_code({"base": "logreg", "scale": True}, "classification")
    prog = Program(code=code, source="seed", label="scale+logreg")

    job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xse, kind="classification",
              backend=SKLEARN_SPEC, limits=ResourceLimits(wall_seconds=40, cpu_seconds=35))
    res = LocalSubprocessExecutor().run(job)
    assert res.ok, f"sealed run failed: {res.error_kind} {res.error}"
    assert not hasattr(res, "score"), "RunResult must not carry a score (firewall)"

    cert = certify.certify_on_sealed(task, s, res.preds)
    assert "lower_bound" in cert and cert["peeks"] == 1, "exactly one sealed peek for the winner"
    assert cert["lower_bound"] <= cert["observed"] + 1e-9
    print(f"[ok] executor preds certified on sealed: lb={cert['lower_bound']:.4f} "
          f"theta={cert['theta']} certified={cert['certified']} peeks={cert['peeks']}")


def test_torch_probe_degrades_honestly():
    """No torch here => TORCH_SPEC not runnable, with an honest reason; never faked."""
    cap_sk = probe_backend(SKLEARN_SPEC)
    cap_t = probe_backend(TORCH_SPEC)
    assert cap_sk.runnable, "sklearn must be runnable in jax-env-311"
    try:
        import torch  # noqa: F401
        torch_present = True
    except Exception:
        torch_present = False
    if torch_present:
        assert cap_t.runnable, "torch present => probe should report runnable"
        print(f"[ok] torch present; probe runnable cuda={cap_t.cuda} devices={cap_t.devices}")
    else:
        assert not cap_t.runnable, "torch absent => probe must report NOT runnable"
        assert "torch" in cap_t.reason, f"reason must name the missing module: {cap_t.reason}"
        print(f"[ok] torch absent; honest decline reason: {cap_t.reason!r}")


def test_torch_tagged_program_declines_not_reroutes():
    """A torch-tagged Program must return backend_unavailable, NOT a silent sklearn reroute."""
    try:
        import torch  # noqa: F401
        print("[skip] torch present; the decline path is for the no-torch machine")
        return
    except Exception:
        pass
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    # Note: code defines build_module (torch entrypoint), not build_estimator — proving no reroute.
    prog = Program(code="def build_module(n_in, n_out, kind):\n    raise RuntimeError('unreached')\n",
                   source="llm", label="torch_mlp")
    job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
              backend=TORCH_SPEC, limits=ResourceLimits(wall_seconds=20, cpu_seconds=15),
              fit_spec=FitSpec(epochs=1))
    res = LocalSubprocessExecutor().run(job)
    assert not res.ok, "torch arm must not succeed on a machine with no torch"
    assert res.error_kind == "backend_unavailable", \
        f"expected backend_unavailable, got {res.error_kind!r} ({res.error})"
    assert "torch" in res.error
    print(f"[ok] torch arm declined honestly: [{res.error_kind}] {res.error[:70]}")


def test_sklearn_build_error_is_typed():
    """A broken candidate surfaces as a typed error, not a parent crash (Phase-0 taxonomy kept)."""
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    bad = Program(code="def build_estimator():\n    raise RuntimeError('boom')\n",
                  source="seed", label="broken")
    job = Job(program=bad, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
              backend=SKLEARN_SPEC, limits=ResourceLimits(wall_seconds=20, cpu_seconds=15))
    res = LocalSubprocessExecutor().run(job)
    assert not res.ok and res.error_kind in ("build", "fit", "other")
    print(f"[ok] sklearn build error typed: [{res.error_kind}] {res.error[:60]}")


class _FakeProvider:
    """Duck-typed stand-in for a vfplatform GPU provider. Runs the bundle LOCALLY via the same
    runner contract (acquire/push/exec/pull/release), proving GPU is a pure substrate swap."""

    def __init__(self, fail_acquire=False):
        self.fail_acquire = fail_acquire
        self.released = False
        self._dir = None

    def acquire(self, *, gpu, image):
        if self.fail_acquire:
            raise RuntimeError("no capacity")
        return {"gpu": gpu, "image": image}

    def push(self, handle, local_dir, remote):
        self._dir = local_dir  # local stand-in: the bundle is already on disk here

    def exec(self, handle, cmd, timeout):
        # Mimic `cd /work && timeout N python runner.py <args>` by running the LOCAL bundle dir.
        import shlex
        import subprocess as sp
        toks = shlex.split(cmd)
        i = toks.index("runner.py")
        args = toks[i + 1:]
        argv = [sys.executable, os.path.join(self._dir, "runner.py")]
        for a in args:
            argv.append(os.path.join(self._dir, a) if not a in ("classification", "regression",
                                                                 "sklearn", "torch") else a)
        p = sp.run(argv, capture_output=True, text=True, timeout=timeout)
        last = (p.stdout or "").strip().splitlines()
        return {"status": last[-1] if last else "", "stdout": p.stdout,
                "stderr": p.stderr, "returncode": p.returncode}

    def pull(self, handle, remote, local):
        pass  # local stand-in: files are already where parse_runner_result expects them

    def release(self, handle):
        self.released = True


def test_remote_pod_executor_roundtrips_same_contract():
    """RemotePodExecutor round-trips the SAME runner contract through a fake provider (sklearn)."""
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    code = make_code({"base": "logreg", "scale": True}, "classification")
    prog = Program(code=code, source="seed", label="scale+logreg")

    ref = sandbox.run_program(prog, Xtr, ytr, Xva, kind="classification", wall_seconds=40)
    prov = _FakeProvider()
    rex = RemotePodExecutor(prov, image="frontier:cpu", gpu="A100", remote_dir="/work")
    job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
              backend=SKLEARN_SPEC, limits=ResourceLimits(wall_seconds=40, cpu_seconds=35))
    res = rex.run(job)
    assert res.ok, f"remote run failed: {res.error_kind} {res.error}"
    assert [str(a) for a in res.preds] == [str(b) for b in ref.preds], \
        "remote-path preds must match local Phase-0 preds (identical bundle+runner+parser)"
    assert prov.released, "pod must be released in finally (cost cap)"
    print(f"[ok] RemotePodExecutor round-trips identical contract; pod released; "
          f"{len(res.preds)} preds match local")


def test_remote_acquire_failure_is_typed_remote():
    """An infra failure (acquire) is error_kind='remote', separable from a bad candidate."""
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    code = make_code({"base": "logreg"}, "classification")
    prog = Program(code=code, source="seed", label="logreg")
    rex = RemotePodExecutor(_FakeProvider(fail_acquire=True), image="x", gpu="A100")
    job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
              backend=SKLEARN_SPEC, limits=ResourceLimits(wall_seconds=20, cpu_seconds=15))
    res = rex.run(job)
    assert not res.ok and res.error_kind == "remote", f"expected remote, got {res.error_kind}"
    print(f"[ok] remote acquire failure typed: [{res.error_kind}] {res.error[:60]}")


def test_cost_model_prefers_cpu_for_small_problems():
    """Cost model: sklearn run is $0; CUDA torch costs dollars and is slower per the rough model."""
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    code = make_code({"base": "logreg"}, "classification")
    prog = Program(code=code, source="seed", label="logreg")
    sk_job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
                 backend=SKLEARN_SPEC, limits=ResourceLimits(), fit_spec=None)
    t_job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
                backend=TORCH_SPEC, limits=ResourceLimits(), fit_spec=FitSpec(epochs=100))
    from frontier.core.execution import Capability
    c_sk = cost_model(sk_job, Capability("sklearn", runnable=True))
    c_gpu = cost_model(t_job, Capability("torch", runnable=True, cuda=True))
    assert c_sk.dollars == 0.0, "sklearn local run must be $0"
    assert c_gpu.dollars > 0.0, "a CUDA run must carry an estimated dollar cost"
    print(f"[ok] cost model: sklearn=${c_sk.dollars:.4f}/{c_sk.seconds:.3f}s, "
          f"gpu=${c_gpu.dollars:.4f}/{c_gpu.seconds:.3f}s")


def test_bundle_byte_compatible_npz_for_sklearn():
    """The sklearn bundle npz has EXACTLY the Phase-0 keys/dtypes (no extras) => identical bytes path."""
    import tempfile
    task = _clf_task()
    _, Xtr, ytr, Xva, _ = _splits_and_arrays(task)
    code = make_code({"base": "logreg"}, "classification")
    prog = Program(code=code, source="seed", label="logreg")
    job = Job(program=prog, X_train=Xtr, y_train=ytr, X_eval=Xva, kind="classification",
              backend=SKLEARN_SPEC, limits=ResourceLimits())
    with tempfile.TemporaryDirectory() as d:
        paths = serialize_job_to_dir(job, d)
        z = np.load(paths.job, allow_pickle=True)
        assert sorted(z.files) == ["Xev", "Xtr", "ytr"], \
            f"sklearn npz must carry only Phase-0 keys, got {sorted(z.files)}"
        assert z["Xtr"].dtype == np.float64 and z["ytr"].dtype == object
        # And the runner written is the verbatim Phase-0 runner.
        assert open(paths.runner).read() == sandbox._RUNNER, "sklearn runner must be byte-identical"
    print("[ok] sklearn bundle npz/runner byte-compatible with Phase-0")


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
    print(f"\n{len(fns) - failed}/{len(fns)} execution-substrate tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
