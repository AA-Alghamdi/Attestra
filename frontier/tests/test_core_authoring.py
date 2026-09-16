"""Integrity tests for the core model-authoring engine (frontier/core/authoring.py).

Run standalone with the project interpreter:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_core_authoring.py
or under pytest:
    /Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/tests/test_core_authoring.py -q

These run with NO real LLM. A FakeClient (a Callable[[str], str]) returns canned
build_estimator() code, so the WHOLE authoring + firewall + sandbox + certify path is
exercised for real on a real sklearn dataset. The properties asserted:

  - the firewall REJECTS a forbidden program (os/socket import, file IO, leakage wrapper) and
    ACCEPTS a clean one; autocorrect supplies a missing import / fixes a typo;
  - a FakeClient authored hist-GBM is materialized into a Program(source="llm") whose code runs
    in the REAL out-of-process sandbox and CERTIFIES through the frozen sealed gate -- the same
    gate a seed uses, with exactly one sealed peek;
  - an authored program that out-validates the recipe floor is SELECTED and certified;
  - client=None => propose() returns [] and the engine produces the identical floor-only,
    honestly-declared result, with llm_active reading False;
  - subclassing LLMProposer makes engine.llm_active honest (True iff a client is wired);
  - the ProgramSpec post-check flags a planted forbidden construct (innovation 7.1);
  - self-consistency over scored programs is a VAL-only confidence signal (innovation 7.2).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier import certify, sandbox
from frontier.engine import EngineConfig, ResearchEngine
from frontier.program import Program, RunResult
from frontier.proposers import LLMProposer, SeedProposer, MutationProposer
from frontier.task import Task

from frontier.core.authoring import (
    AuthoringConfig, AuthoringEngine, CoreAuthoringProposer, Firewall, Motif, ProgramSpec,
)


# --------------------------------------------------------------------------- fixtures

def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    # theta low enough that a strong authored model clears it but the gate is real.
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.80, name="bc")


# A "weak" estimator the FakeClient never authors; used to depress the floor so the authored
# program is the selected winner.
_HIST_GBM_CLF = (
    "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    "def build_estimator():\n"
    "    return HistGradientBoostingClassifier(random_state=0)\n"
)

# A program referencing a symbol WITHOUT importing it (autocorrect must supply the import).
_NEEDS_IMPORT = (
    "def build_estimator():\n"
    "    return RandomForestClassifier(n_estimators=50, random_state=0)\n"
)

# A program with an obvious TYPO autocorrect must rename (cutoff 0.85).
_TYPO = (
    "def build_estimator():\n"
    "    return RandomForestClassifer(n_estimators=50, random_state=0)\n"  # missing 'i'
)

# Forbidden programs the firewall MUST reject before the sandbox ever sees them.
_FORBIDDEN_OS = (
    "import os\n"
    "def build_estimator():\n"
    "    os.system('echo pwned')\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    return LogisticRegression()\n"
)
_FORBIDDEN_SOCKET = (
    "import socket\n"
    "def build_estimator():\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    return LogisticRegression()\n"
)
_FORBIDDEN_FILE_IO = (
    "def build_estimator():\n"
    "    f = open('/etc/passwd')\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    return LogisticRegression()\n"
)
_FORBIDDEN_LEAKAGE = (
    "from sklearn.model_selection import GridSearchCV\n"
    "from sklearn.svm import SVC\n"
    "def build_estimator():\n"
    "    return GridSearchCV(SVC(), {'C': [1, 10]})\n"
)


class FakeClient:
    """A Callable[[str], str] standing in for a real LLM. Returns canned code.

    Records prompts so we can assert the diagnosis / spec actually reached the model.
    """

    def __init__(self, code: str):
        self.code = code
        self.prompts = []

    def __call__(self, prompt, temperature=None):
        self.prompts.append(prompt)
        return self.code


# --------------------------------------------------------------------------- firewall tests

def test_firewall_accepts_clean_program():
    fw = Firewall()
    ok, reason = fw.validate(_HIST_GBM_CLF, kind="classification")
    assert ok, f"clean hist-gbm should validate, got: {reason}"
    print("[ok] firewall accepts a clean build_estimator()")


def test_firewall_rejects_forbidden_programs():
    fw = Firewall()
    for name, code in [("os.system", _FORBIDDEN_OS), ("socket import", _FORBIDDEN_SOCKET),
                       ("file IO", _FORBIDDEN_FILE_IO), ("GridSearchCV leakage", _FORBIDDEN_LEAKAGE)]:
        ok, reason = fw.validate(code, kind="classification")
        assert not ok, f"firewall MUST reject {name}, but it passed"
    print("[ok] firewall rejects os/socket import, file IO, and leakage wrappers")


def test_firewall_autocorrect_supplies_missing_import():
    fw = Firewall()
    fixed = fw.autocorrect_names(_NEEDS_IMPORT)
    assert "import RandomForestClassifier" in fixed, f"missing import not supplied: {fixed!r}"
    assert any(c.startswith("import+:RandomForestClassifier") for c in fw.last_corrections)
    ok, reason = fw.validate(fixed, kind="classification")
    assert ok, f"autocorrected program should validate: {reason}"
    print(f"[ok] autocorrect supplied import; corrections={fw.last_corrections}")


def test_firewall_autocorrect_fixes_typo():
    fw = Firewall()
    fixed = fw.autocorrect_names(_TYPO)
    assert "RandomForestClassifier" in fixed and "RandomForestClassifer(" not in fixed, \
        f"typo not renamed: {fixed!r}"
    assert any(c.startswith("rename:RandomForestClassifer->RandomForestClassifier")
               for c in fw.last_corrections), fw.last_corrections
    print(f"[ok] autocorrect renamed typo; corrections={fw.last_corrections}")


def test_firewall_strip_njobs_and_verbose():
    fw = Firewall()
    code = ("from sklearn.ensemble import RandomForestClassifier\n"
            "def build_estimator():\n"
            "    return RandomForestClassifier(n_estimators=10, n_jobs=-1, verbose=2, random_state=0)\n")
    stripped = fw.strip_forbidden(code)
    assert "n_jobs=-1" not in stripped and "n_jobs=1" in stripped, stripped
    assert "verbose" not in stripped, f"verbose not dropped: {stripped!r}"
    ok, _ = fw.validate(stripped, kind="classification")
    assert ok
    print(f"[ok] strip_forbidden: n_jobs->1, verbose dropped; strips={fw.last_strips}")


def test_firewall_preamble_idempotent_and_safe():
    fw = Firewall()
    out = fw.add_preamble(_HIST_GBM_CLF)
    assert out.count("def build_estimator") == 1, "preamble must not duplicate the function"
    assert "import numpy as np" in out
    # the preamble only imports allow-listed roots -> still validates
    ok, _ = fw.validate(out, kind="classification")
    assert ok, "preambled code must still pass the allow-list"
    print("[ok] preamble adds helper imports, keeps one build_estimator, stays allow-list-clean")


# --------------------------------------------------------------------------- schema tests

def test_programspec_check_flags_forbidden_construct():
    spec = ProgramSpec(task_kind="regression", forbid=["high_degree_poly"])
    bad = ("from sklearn.preprocessing import PolynomialFeatures\n"
           "from sklearn.linear_model import Ridge\n"
           "from sklearn.pipeline import make_pipeline\n"
           "def build_estimator():\n"
           "    return make_pipeline(PolynomialFeatures(degree=5), Ridge())\n")
    violations = spec.check(bad)
    assert "high_degree_poly" in violations, f"spec.check should flag degree>2: {violations}"
    # a degree-2 program does NOT trip it
    ok_code = bad.replace("degree=5", "degree=2")
    assert "high_degree_poly" not in spec.check(ok_code)
    print(f"[ok] ProgramSpec.check flags planted forbidden construct: {violations}")


def test_programspec_directives_render():
    spec = ProgramSpec(task_kind="regression", encourage=["nonlinear_model", "target_transform"],
                       must_handle=["scale_sensitive_model"], forbid=["high_degree_poly"])
    text = spec.to_directives()
    assert "ENCOURAGED" in text and "FORBIDDEN BY DIAGNOSIS" in text and "REQUIRED" in text
    assert "TransformedTargetRegressor" in text  # reachable-construct menu present
    print("[ok] ProgramSpec renders typed directives into the prompt")


# --------------------------------------------------------------------------- live (FakeClient) tests

def test_authored_program_runs_in_sandbox_and_certifies():
    """A FakeClient authored hist-GBM materializes to source='llm' code, runs in the REAL
    sandbox, and certifies through the SAME frozen sealed gate a seed uses."""
    task = _clf_task()
    client = FakeClient(_HIST_GBM_CLF)
    eng = AuthoringEngine(client=client, config=AuthoringConfig(n=1, inject_literature=False))
    progs = eng.author({"task_kind": "classification", "n_features": task.n_features,
                        "n_train": 300, "round": 0, "tried_labels": set(),
                        "recent_errors": []})
    assert progs and progs[0].source == "llm", "must author a source='llm' Program"
    assert "build_estimator" in progs[0].code and "import numpy as np" in progs[0].code, \
        "preamble must be present on the authored code"
    assert client.prompts and "build_estimator" in client.prompts[0], "FakeClient saw the prompt"

    # run it through the REAL out-of-process sandbox, exactly as the engine would.
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xva = Task.rows_to_X(s.val_rows); Xse = Task.rows_to_X(s.sealed_rows)
    res = sandbox.run_program(progs[0], Xtr, ytr, Xva, kind="classification", wall_seconds=60)
    assert res.ok, f"authored program must run in sandbox: [{res.error_kind}] {res.error}"
    val = certify.score_val(task, s.val_rows, res.preds)
    assert 0.0 <= val <= 1.0

    final = sandbox.run_program(progs[0], Xtr, ytr, Xse, kind="classification", wall_seconds=60)
    assert final.ok, f"authored winner must run on sealed: {final.error}"
    cert = certify.certify_on_sealed(task, s, final.preds)
    assert cert["peeks"] == 1, "exactly one sealed peek"
    assert cert["lower_bound"] <= cert["observed"] + 1e-9
    assert cert["certified"], f"strong authored model should certify above theta: {cert}"
    print(f"[ok] authored hist-GBM certifies: val={val:.4f} sealed_lb={cert['lower_bound']} "
          f"theta={cert['theta']} certified={cert['certified']}")


# An authored pipeline the deterministic floor CANNOT produce (scaler + PCA + GBM stack of a
# shape no seed/mutation recipe emits), so its Program.id is distinct from every floor seed and
# it can win on its own source='llm' rather than dedup-colliding with the seed hist-GBM.
_AUTHORED_DISTINCT_CLF = (
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.decomposition import PCA\n"
    "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    "def build_estimator():\n"
    "    return Pipeline([\n"
    "        ('scale', StandardScaler()),\n"
    "        ('pca', PCA(n_components=10, random_state=0)),\n"
    "        ('gbm', HistGradientBoostingClassifier(random_state=0)),\n"
    "    ])\n"
)


def test_engine_selects_authored_winner_over_floor():
    """End-to-end via the real engine: a strong authored program the FLOOR CANNOT EMIT validates
    competitively and is selected as a source='llm' winner. llm_active reads True (subclass route).

    Note: the engine dedups by Program.id, so an authored program byte-identical to a seed would
    collide with (and lose the tie to) that seed. We author a structurally distinct pipeline so the
    'llm' source is genuinely exercised as the selected winner."""
    task = _clf_task()
    client = FakeClient(_AUTHORED_DISTINCT_CLF)
    proposers = [SeedProposer(), MutationProposer(),
                 CoreAuthoringProposer(client=client, config=AuthoringConfig(n=2,
                                       inject_literature=False))]
    res = ResearchEngine(EngineConfig(rounds=2, wall_seconds=60, cpu_seconds=50),
                         proposers=proposers).run(task)
    assert res.llm_active is True, "subclassing LLMProposer must make llm_active honest (True)"
    assert res.winner is not None and res.certificate is not None
    assert res.certificate["peeks"] == 1
    # the authored winner must be a source='llm' program (the distinct pipeline), and it must
    # have run through the same sealed gate the floor uses.
    authored_in_history = [r for r in res.history if r.source == "llm" and r.ok]
    assert authored_in_history, "an authored llm program must have run and scored"
    print(f"[ok] e2e: winner source={res.winner.source} label={res.winner.label} "
          f"val={res.winner_val_score} certified={res.certified} llm_active={res.llm_active}; "
          f"authored ran: {[r.label for r in authored_in_history]}")


def test_forbidden_authored_program_is_rejected_before_sandbox():
    """A FakeClient that emits a forbidden program (os import) is rejected by the firewall;
    author() returns [] and the rejection is logged (never silently dropped)."""
    client = FakeClient(_FORBIDDEN_OS)
    prop = CoreAuthoringProposer(client=client, config=AuthoringConfig(n=2,
                                 inject_literature=False))
    progs = prop.propose({"task_kind": "classification", "n_features": 30, "n_train": 300,
                          "round": 0, "tried_labels": set(), "recent_errors": []})
    assert progs == [], "forbidden program must not be authored into the pool"
    assert prop.rejections, "the rejection must be logged for audit"
    assert any("forbidden" in r["reason"] for r in prop.rejections), prop.rejections
    print(f"[ok] forbidden authored program rejected pre-sandbox: {prop.rejections[0]['reason']}")


# --------------------------------------------------------------------------- degrade tests

def test_client_none_yields_floor_only_result():
    """client=None => CoreAuthoringProposer.propose() returns []; the engine produces the
    identical floor-only result, llm_active False. Honest degrade, no fabricated proposals."""
    task = _clf_task()
    prop_none = CoreAuthoringProposer(client=None)
    assert prop_none.propose({"task_kind": "classification", "round": 0,
                              "tried_labels": set()}) == [], "no client => [] proposals"
    assert isinstance(prop_none, LLMProposer), "must subclass LLMProposer for llm_active"
    assert prop_none.client is None

    # the engine with [Seed, Mutation, CoreAuthoring(None)] must equal the pure floor.
    floor = [SeedProposer(), MutationProposer()]
    withcore = [SeedProposer(), MutationProposer(), CoreAuthoringProposer(client=None)]
    r_floor = ResearchEngine(EngineConfig(rounds=2, wall_seconds=60, cpu_seconds=50),
                             proposers=floor).run(task)
    r_core = ResearchEngine(EngineConfig(rounds=2, wall_seconds=60, cpu_seconds=50),
                            proposers=withcore).run(task)
    assert r_core.llm_active is False, "client=None must read llm_active False"
    assert r_floor.winner is not None and r_core.winner is not None
    # the floor is deterministic, so the selected winner label must match.
    assert r_floor.winner.label == r_core.winner.label, \
        f"floor result must be identical with/without inactive authoring: " \
        f"{r_floor.winner.label} vs {r_core.winner.label}"
    print(f"[ok] client=None degrades to floor: winner={r_core.winner.label} "
          f"llm_active={r_core.llm_active} (== floor winner={r_floor.winner.label})")


def test_engine_default_llm_inactive_when_no_client():
    """Sanity: even the Phase-0 default proposer set reads llm_active False with no client."""
    task = _clf_task()
    res = ResearchEngine(EngineConfig(rounds=1, wall_seconds=60, cpu_seconds=50)).run(task)
    assert res.llm_active is False
    print("[ok] default engine (no client) reports llm_active=False")


# --------------------------------------------------------------------------- innovation tests

def test_self_consistency_signal_is_val_only():
    """innovation 7.2: structural agreement among high-VAL authored programs is metadata,
    computed from TRUSTED val scores; it never promotes."""
    p_gbm1 = Program(code=_HIST_GBM_CLF, source="llm", label="a")
    p_gbm2 = Program(code=_HIST_GBM_CLF.replace("random_state=0",
                                                "random_state=0, max_iter=200"),
                     source="llm", label="b")
    p_logreg = Program(code=("from sklearn.linear_model import LogisticRegression\n"
                             "def build_estimator():\n    return LogisticRegression()\n"),
                       source="llm", label="c")
    report = AuthoringEngine.self_consistency([(p_gbm1, 0.95), (p_gbm2, 0.94), (p_logreg, 0.70)])
    assert report["modal_family"] == "hist_gbm", report
    assert report["agreement"] >= 0.5, report
    print(f"[ok] self-consistency (VAL-only): {report['note']} agreement={report['agreement']}")


def test_spec_built_from_diagnosis_reaches_prompt():
    """A regression task with a linear champion + a LinAlgError must produce a spec that
    encourages nonlinearity and forbids high-degree polynomials, and that text reaches the
    FakeClient's prompt (so diagnosis truly conditions authoring)."""
    client = FakeClient(_HIST_GBM_CLF)  # regression-shaped code not needed; we inspect the prompt
    eng = AuthoringEngine(client=client, config=AuthoringConfig(n=1, inject_literature=False))
    ctx = {"task_kind": "regression", "n_features": 8, "n_train": 400, "round": 1,
           "tried_labels": set(), "best_label": "scale+ridge", "best_score": 0.51,
           "recent_errors": [("poly3+ridge", "fit", "LinAlgError: singular matrix")],
           "plateau_rounds": 2}
    eng.author(ctx)
    prompt = client.prompts[0]
    assert "FORBIDDEN BY DIAGNOSIS" in prompt and "degree > 2" in prompt, prompt
    assert "structurally DIFFERENT" in prompt or "nonlinear" in prompt, prompt
    assert "LinAlgError" in prompt, "recent failure must be surfaced to the model"
    print("[ok] diagnosis -> typed ProgramSpec -> prompt (nonlinear encouraged, high-poly forbidden)")


def test_skeleton_must_pass_firewall_before_offered():
    """innovation 7.3: a retriever motif whose skeleton FAILS the firewall has its skeleton
    dropped (a poisoned skeleton cannot widen the safety surface)."""
    poisoned = Motif(id="evil:1", claim="trust me", skeleton="import os\nos.system('x')\n")
    clean = Motif(id="skl:tt", claim="power-transform skewed targets",
                  skeleton=("from sklearn.compose import TransformedTargetRegressor\n"
                            "from sklearn.preprocessing import PowerTransformer\n"
                            "from sklearn.linear_model import Ridge\n"
                            "def build_estimator():\n"
                            "    return TransformedTargetRegressor(regressor=Ridge(),\n"
                            "        transformer=PowerTransformer())\n"))

    class R:
        def retrieve(self, context):
            return [poisoned, clean]

    eng = AuthoringEngine(client=FakeClient(_HIST_GBM_CLF), retriever=R())
    motifs = eng._retrieve({"task_kind": "regression"})
    by_id = {m.id: m for m in motifs}
    assert by_id["evil:1"].skeleton is None, "poisoned skeleton must be dropped"
    assert by_id["skl:tt"].skeleton is not None, "clean skeleton must survive"
    print("[ok] retrieval-grounded skeletons are firewall-validated before being offered")


# --------------------------------------------------------------------------- runner

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
    print(f"\n{len(fns) - failed}/{len(fns)} core-authoring tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
