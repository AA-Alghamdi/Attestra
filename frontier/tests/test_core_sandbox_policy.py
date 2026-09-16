"""Integrity tests for frontier.core.sandbox_policy.

What these assert (design 05 acceptance criteria):
  * tier selection + EnforcedGuarantees STAMPING reflect what actually held on THIS OS (not what was
    requested) -- the honesty contract H1/H2/H4;
  * untrusted-on-LOCAL warns, and strict mode REFUSES;
  * the advisory AST checker flags a forbidden import (and a numpy-I/O attr, and the missing entrypoint);
  * the env scrubber strips secrets and applies the BLAS caps + scratch redirection;
  * predictions validation rejects NaN / wrong-length / unknown-label vectors;
  * a REAL local dispatch through the FROZEN Phase-0 runner on a real sklearn dataset returns predictions
    only (firewall) and is stamped with the LOCAL EnforcedGuarantees probed for this host;
  * a registered fake T2 substrate is dispatched to and stamped honestly (no torch / no Linux needed).

Run standalone:
    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_core_sandbox_policy.py
or:
    /Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/tests/test_core_sandbox_policy.py -q

No torch and no Linux required: the LOCAL path runs for real on a tiny sklearn dataset, and the higher
tiers are exercised via a registered fake substrate + the probe (the design's T2/T3 real substrates are
separate modules; this policy layer is OS-agnostic and tests what the probe says THIS host can enforce).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Insert repo root on sys.path, exactly like frontier/tests/test_spine.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import sandbox as spine_sandbox          # the FROZEN Phase-0 runner (we inject it)
from frontier.program import Program
from frontier.core import sandbox_policy as sp
from frontier.core.sandbox_policy import (
    Tier, SandboxPolicy, StrictRefusal, probe_host, ast_check, scrub_env,
    validate_predictions, enforced_of, register_substrate, t2_status, t3_status,
)


# --------------------------------------------------------------------- probe + capability honesty
def test_probe_matches_this_os():
    caps = probe_host()
    assert caps.platform == sys.platform
    assert caps.posix == (os.name == "posix")
    # The load-bearing honesty bit: RLIMIT_AS is NOT enforced on Darwin even though the symbol exists.
    if sys.platform == "darwin":
        assert caps.rlimit_as_enforced is False, "macOS must report RLIMIT_AS unenforced"
    print(f"[ok] probe reflects this host: {caps.as_dict()}")


# --------------------------------------------------------------------- tier resolution + degradation
def test_local_request_resolves_local_and_stamps_this_os():
    caps = probe_host()
    pol = SandboxPolicy(Tier.LOCAL, untrusted=False, local_runner=spine_sandbox.run_program, caps=caps)
    resolved, notes = pol.resolve()
    assert resolved == Tier.LOCAL
    g = pol._probe_enforced(resolved, notes, ast_ran=True)
    # the stamp must reflect the probe, not a wish:
    expect_mem = "rlimit-as" if caps.rlimit_as_enforced else "wall-timeout-only"
    assert g.mem_guard == expect_mem, f"mem_guard {g.mem_guard} != probed {expect_mem}"
    assert g.network == "not-isolated"        # Tier-1 stock host: NIC is live, honestly
    assert g.fs == "tempdir-cwd"
    assert g.uid_drop is False and g.no_new_privs is False and g.seccomp == "unavailable"
    assert g.wall_timeout is True
    print(f"[ok] LOCAL stamp reflects this OS: mem_guard={g.mem_guard} network={g.network}")


def test_higher_tier_degrades_downward_when_unavailable():
    # Request CONTAINER with NO substrate registered -> must degrade DOWNWARD (never claim it ran).
    sp._SUBSTRATES.clear()
    caps = probe_host()
    pol = SandboxPolicy(Tier.CONTAINER, untrusted=False, local_runner=spine_sandbox.run_program, caps=caps)
    resolved, notes = pol.resolve()
    assert sp._RANK[resolved] <= sp._RANK[Tier.CONTAINER], "must not upgrade past requested"
    assert resolved != Tier.CONTAINER, "container with no substrate must not be claimed"
    assert notes, "a degradation must be announced in notes"
    g = pol._probe_enforced(resolved, notes, ast_ran=False)
    assert g.tier == resolved, "stamp tier == resolved tier, not requested"
    assert any("degrade" in n for n in g.notes)
    print(f"[ok] CONTAINER->{resolved.value} degraded, notes={list(g.notes)}")


def test_registered_fake_t2_substrate_is_dispatched_and_stamped():
    # A fake T2 substrate (same signature as run_program) lets us exercise tier!=LOCAL dispatch + stamping
    # WITHOUT a Linux host. It is honest: the policy still stamps the T2 guarantees the design defines, and
    # the test asserts the *structure* (uid_drop True, network per t2_status), not a real kernel enforcement.
    sp._SUBSTRATES.clear()
    calls = {}

    def fake_t2(program, X_train, y_train, X_eval, *, kind, wall_seconds, cpu_seconds, address_mb):
        calls["hit"] = True
        from frontier.program import RunResult
        return RunResult(program.id, ok=True, preds=[str(y_train[0])] * len(X_eval), wall_seconds=0.01)

    # Make t2 'available' for the resolver by registering the substrate AND faking the host caps to linux+setpriv.
    caps = probe_host()
    linuxish = sp.HostCapabilities(
        platform="linux", posix=True, rlimit_cpu=True, rlimit_as_enforced=True, rlimit_nproc=True,
        has_setpriv=True, has_unshare=False, has_seccomp=False, has_docker=False, has_gpu=False)
    register_substrate(Tier.LINUX_UID, fake_t2)
    pol = SandboxPolicy(Tier.LINUX_UID, untrusted=True, local_runner=spine_sandbox.run_program, caps=linuxish)
    resolved, notes = pol.resolve()
    assert resolved == Tier.LINUX_UID, f"with substrate+caps, T2 must hold; got {resolved}"
    prog = Program(code="def build_estimator():\n    pass\n", source="seed", label="noop")
    res = pol.run(prog, np.zeros((3, 2)), np.array(["a", "b", "a"]), np.zeros((4, 2)), kind="classification")
    assert calls.get("hit"), "the registered T2 substrate must be dispatched to"
    g = enforced_of(res)
    assert g.tier == Tier.LINUX_UID and g.uid_drop is True and g.no_new_privs is True
    # netns absent + seccomp absent on the faked host -> honest "not-isolated" network (R4), NOT a false "netns".
    assert g.network == "not-isolated", f"expected honest not-isolated, got {g.network}"
    sp._SUBSTRATES.clear()
    print(f"[ok] fake T2 dispatched + stamped honestly: tier={g.tier.value} network={g.network}")


def test_untrusted_on_local_warns_and_strict_refuses():
    sp._SUBSTRATES.clear()
    # non-strict: warns but runs (resolve returns LOCAL with a note)
    pol = SandboxPolicy(Tier.LOCAL, untrusted=True, strict=False,
                        local_runner=spine_sandbox.run_program)
    resolved, notes = pol.resolve()
    assert resolved == Tier.LOCAL
    assert any("untrusted-on-LOCAL" in n for n in notes), "must record the loud-warning note"
    # strict: refuses
    pol2 = SandboxPolicy(Tier.LOCAL, untrusted=True, strict=True,
                         local_runner=spine_sandbox.run_program)
    raised = False
    try:
        pol2.resolve()
    except StrictRefusal:
        raised = True
    assert raised, "strict mode must REFUSE untrusted-on-LOCAL"
    print("[ok] untrusted-on-LOCAL warns (non-strict) and refuses (strict)")


# --------------------------------------------------------------------- advisory AST checker
def test_ast_checker_flags_forbidden_import():
    code = "import os\ndef build_estimator():\n    return os\n"
    rep = ast_check(code)
    assert not rep.ok
    assert any("os" in v for v in rep.violations), rep.violations
    print(f"[ok] AST flags forbidden import: {rep.violations}")


def test_ast_checker_flags_numpy_io_and_socket_and_missing_entrypoint():
    # numpy file-I/O attr (the confirmed exfil vector) with no banned import name:
    code = "import numpy as np\ndef build_estimator():\n    np.save('/tmp/x', np.zeros(3))\n"
    rep = ast_check(code)
    assert not rep.ok and any("save" in v for v in rep.violations), rep.violations
    # socket import:
    rep2 = ast_check("import socket\ndef build_estimator():\n    return 1\n")
    assert not rep2.ok and any("socket" in v for v in rep2.violations)
    # missing entrypoint:
    rep3 = ast_check("import numpy as np\nx = 1\n")
    assert not rep3.ok and any("build_estimator" in v for v in rep3.violations)
    print("[ok] AST flags numpy-I/O attr, socket import, and missing entrypoint")


def test_ast_checker_passes_a_clean_estimator():
    code = (
        "from sklearn.linear_model import LogisticRegression\n"
        "def build_estimator():\n"
        "    return LogisticRegression(max_iter=200)\n"
    )
    rep = ast_check(code)
    assert rep.ok, f"clean estimator should pass: {rep.violations}"
    assert rep.has_entrypoint
    print("[ok] AST passes a clean sklearn estimator")


def test_policy_rejects_forbidden_code_without_executing():
    sp._SUBSTRATES.clear()
    ran = {"hit": False}

    def tripwire(*a, **k):
        ran["hit"] = True
        from frontier.program import RunResult
        return RunResult("x", ok=True)

    pol = SandboxPolicy(Tier.LOCAL, untrusted=True, local_runner=tripwire)
    bad = Program(code="import socket\ndef build_estimator():\n    return socket\n",
                  source="llm", label="evil")
    res = pol.run(bad, np.zeros((2, 2)), np.array(["a", "b"]), np.zeros((2, 2)), kind="classification")
    assert not res.ok and res.error_kind == "policy", (res.ok, res.error_kind)
    assert ran["hit"] is False, "forbidden code must NOT be dispatched to the runner"
    assert enforced_of(res) is not None, "even a policy-reject is stamped with what would have held"
    print(f"[ok] policy reject before execution: [{res.error_kind}] {res.error[:70]}")


# --------------------------------------------------------------------- env scrubber
def test_env_scrubber_strips_secrets():
    dirty = {
        "PATH": "/usr/bin:/bin", "HOME": "/Users/x",
        "ANTHROPIC_API_KEY": "sk-secret", "OPENAI_API_KEY": "sk-2", "AWS_SECRET_ACCESS_KEY": "z",
        "GITHUB_TOKEN": "ghp_x", "MY_PASSWORD": "p", "SESSION_COOKIE": "c",
        "RANDOM_NONSENSE": "keep-me-not",   # not on the allow-list -> dropped (allow-list, not deny-list)
        "LC_ALL": "en_US.UTF-8",
    }
    clean = scrub_env(dirty, scratch_dir="/tmp/scratch123")
    # every secret marker gone
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
              "MY_PASSWORD", "SESSION_COOKIE"):
        assert k not in clean, f"{k} leaked into scrubbed env"
    # non-allow-listed (even non-secret) var dropped: allow-list is the safe default
    assert "RANDOM_NONSENSE" not in clean
    # allow-listed kept
    assert clean["PATH"] and clean["LC_ALL"] == "en_US.UTF-8"
    # scratch redirection applied
    assert clean["HOME"] == "/tmp/scratch123" and clean["TMPDIR"] == "/tmp/scratch123"
    # BLAS caps applied (mandatory for CPU accounting)
    assert clean["OMP_NUM_THREADS"] == "1" and clean["MALLOC_ARENA_MAX"] == "2"
    # the source dict is not mutated, os.environ untouched
    assert "ANTHROPIC_API_KEY" in dirty
    # a caller cannot re-inject a secret via extra=
    clean2 = scrub_env({"PATH": "/bin"}, extra={"SNEAKY_TOKEN": "x", "OK_VAR": "y"})
    assert "SNEAKY_TOKEN" not in clean2 and clean2["OK_VAR"] == "y"
    print("[ok] env scrubber strips secrets, applies BLAS caps + scratch redirect, no re-inject")


# --------------------------------------------------------------------- predictions validation (T11)
def test_predictions_validation():
    ok, _ = validate_predictions(["a", "b", "a"], kind="classification", n_expected=3, labels=["a", "b"])
    assert ok
    bad_label, why = validate_predictions(["a", "c"], kind="classification", labels=["a", "b"])
    assert not bad_label and "label" in why
    bad_len, why2 = validate_predictions(["a"], kind="classification", n_expected=3, labels=["a"])
    assert not bad_len and "count" in why2
    nan_ok, why3 = validate_predictions([1.0, float("nan"), 2.0], kind="regression")
    assert not nan_ok and "non-finite" in why3
    fin_ok, _ = validate_predictions([1.0, 2.5, -3.0], kind="regression")
    assert fin_ok
    print("[ok] predictions validation: rejects unknown-label, wrong-length, NaN; accepts finite floats")


# --------------------------------------------------------------------- REAL local dispatch (firewall)
def test_real_local_dispatch_on_dataset_returns_predictions_only():
    """End-to-end through the FROZEN Phase-0 runner on a real sklearn dataset. Asserts the firewall
    (predictions only, no score) AND that the result is stamped with the LOCAL EnforcedGuarantees probed
    for THIS host. This is the sklearn/stand-in path exercised for real (no torch needed)."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    X, y = d.data, d.target.astype(str)
    Xtr, ytr, Xev = X[:400], y[:400], X[400:]

    sp._SUBSTRATES.clear()
    pol = SandboxPolicy(Tier.LOCAL, untrusted=True, strict=False,
                        local_runner=spine_sandbox.run_program, caps=probe_host())
    code = (
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import make_pipeline\n"
        "def build_estimator():\n"
        "    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))\n"
    )
    prog = Program(code=code, source="seed", label="scale+logreg")
    res = pol.run(prog, Xtr, ytr, Xev, kind="classification", wall_seconds=60, cpu_seconds=50)
    assert res.ok, f"clean estimator should run: [{res.error_kind}] {res.error}"
    assert res.preds is not None and len(res.preds) == len(Xev)
    assert not hasattr(res, "score"), "firewall: RunResult must carry no score"
    # predictions are valid labels
    okv, why = validate_predictions(res.preds, kind="classification", n_expected=len(Xev),
                                    labels=sorted(set(y.tolist())))
    assert okv, why
    # stamped with the probed LOCAL guarantees
    g = enforced_of(res)
    assert g is not None and g.tier == Tier.LOCAL
    expect_mem = "rlimit-as" if probe_host().rlimit_as_enforced else "wall-timeout-only"
    assert g.mem_guard == expect_mem
    assert g.advisory_ast_gate is True, "the AST gate ran before dispatch"
    print(f"[ok] real LOCAL dispatch: {len(res.preds)} preds, no score, "
          f"stamped tier={g.tier.value} mem_guard={g.mem_guard}")


def test_t2_t3_status_reports_are_honest_on_this_host():
    caps = probe_host()
    s2 = t2_status(caps)
    s3 = t3_status(caps, gpu=False)
    s3g = t3_status(caps, gpu=True)
    if sys.platform == "darwin":
        assert s2["available"] is False, "T2 cannot be available on macOS (no setpriv/linux)"
    # GPU lane available only if a usable GPU was actually probed
    assert s3g["gpu"] == caps.has_gpu
    print(f"[ok] tier status honest on this host: t2={s2['reason']} t3={s3['reason']} t3gpu={s3g['reason']}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} sandbox_policy tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
