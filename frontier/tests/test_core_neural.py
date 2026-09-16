"""Tests for the neural architecture core (design 03), runnable locally WITHOUT torch.

    /Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/tests/test_core_neural.py -q
or standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_core_neural.py

What is exercised for real with no torch:
  - NeuralSpec validity (valid specs pass; broken specs return the expected violations);
  - render_program(spec).code compile()s, defines build_estimator(), is self-contained;
  - param-count estimate matches the realized sklearn MLP coefficient count for MLP-only specs;
  - the SklearnMLPStandin certifies END-TO-END through the FROZEN sealed gate (full spine);
  - NASProposer / LLMArchitectProposer protocol incl. diagnosis-conditioned mutation branches.

Torch tests are import-gated with skipif and run only on a torch-capable pod; the skip is
reported, never hidden, so a green local run never masquerades as torch-tested.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, sandbox
from frontier.program import Program
from frontier.task import Task
from frontier.core.neural import (
    Block, NeuralSpec, validate, estimate_params,
    NASProposer, LLMArchitectProposer, neural_fit_signal,
    _widen, _narrow, _add_dropout, _add_norm, _add_block, _deepen,
)
from frontier.core.render import render_program, host_backend


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------- spec helpers

def _tabular_clf_spec(out_dim=2):
    return NeuralSpec(modality="tabular", task_kind="classification", in_features=30,
                      out_dim=out_dim,
                      blocks=[Block(kind="mlp", width=64, activation="relu"),
                              Block(kind="residual_mlp", width=64, residual=True)])


# ----------------------------------------------------------------------- validity tests

def test_valid_specs_pass():
    assert validate(_tabular_clf_spec()) == []
    reg = NeuralSpec(modality="tabular", task_kind="regression", in_features=8, out_dim=1,
                     blocks=[Block(kind="mlp", width=32)])
    assert validate(reg) == []
    seq = NeuralSpec(modality="sequence", task_kind="classification", in_features=50, out_dim=2,
                     blocks=[Block(kind="embedding_pool", width=64, vocab_size=1000),
                             Block(kind="attention", width=64, n_heads=4, norm="layer")])
    assert validate(seq) == []
    print("[ok] valid specs across tabular/regression/sequence pass validate")


def test_broken_specs_flagged():
    # regression with out_dim=3
    bad = NeuralSpec(modality="tabular", task_kind="regression", in_features=8, out_dim=3,
                     blocks=[Block(kind="mlp", width=16)])
    assert any("out_dim == 1" in e for e in validate(bad))
    # attention width not divisible by n_heads
    bad2 = NeuralSpec(modality="sequence", task_kind="classification", in_features=10, out_dim=2,
                      blocks=[Block(kind="attention", width=10, n_heads=4)])
    assert any("not divisible" in e for e in validate(bad2))
    # attention on tabular modality
    bad3 = NeuralSpec(modality="tabular", task_kind="classification", in_features=10, out_dim=2,
                      blocks=[Block(kind="attention", width=8, n_heads=4)])
    assert any("only legal for sequence" in e for e in validate(bad3))
    # conv stack that shrinks sequence below length 1
    bad4 = NeuralSpec(modality="sequence", task_kind="classification", in_features=4, out_dim=2,
                      blocks=[Block(kind="conv1d", width=8, kernel_size=10, stride=1)])
    assert any("shrinks sequence length" in e for e in validate(bad4))
    print("[ok] broken specs each return the expected violation")


def test_param_estimate_matches_sklearn_mlp():
    """For an MLP-only spec, estimate_params must match the realized sklearn coefficient count.

    Sanity on the budget math (design 03 §1.1 / §5.2d). sklearn's MLP biases the OUTPUT layer
    with n_classes units (one-hot), so out_dim for the count is the class count.
    """
    spec = NeuralSpec(modality="tabular", task_kind="classification", in_features=20,
                      out_dim=3, blocks=[Block(kind="mlp", width=16), Block(kind="mlp", width=8)])
    est = estimate_params(spec)
    from sklearn.neural_network import MLPClassifier
    X = np.random.RandomState(0).randn(60, 20)
    y = np.array(["a", "b", "c"] * 20)
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1, random_state=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    realized = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    # closed form: 20*16+16 + 16*8+8 + 8*3+3 == realized
    assert est == realized, f"estimate {est} != realized {realized}"
    print(f"[ok] param estimate {est} matches realized sklearn MLP coef count")


# ----------------------------------------------------------------------- render tests

def test_render_compiles_and_is_self_contained():
    for spec in [
        _tabular_clf_spec(),
        NeuralSpec(modality="tabular", task_kind="regression", in_features=8, out_dim=1,
                   blocks=[Block(kind="mlp", width=32), Block(kind="residual_mlp", width=32)]),
        NeuralSpec(modality="sequence", task_kind="classification", in_features=40, out_dim=2,
                   blocks=[Block(kind="embedding_pool", width=32, vocab_size=500),
                           Block(kind="attention", width=32, n_heads=4, norm="layer")]),
        NeuralSpec(modality="image1d", task_kind="classification", in_features=64, out_dim=2,
                   blocks=[Block(kind="conv1d", width=16, kernel_size=3),
                           Block(kind="mlp", width=32)]),
    ]:
        prog = render_program(spec)
        # (a) compiles
        ns = {}
        exec(compile(prog.code, "<candidate>", "exec"), ns)
        # (b) defines a callable build_estimator
        assert callable(ns.get("build_estimator")), "must define callable build_estimator()"
        # (c) self-contained: no IMPORT STATEMENT pulls frontier (comments may mention it).
        import_lines = [ln.strip() for ln in prog.code.splitlines()
                        if ln.strip().startswith(("import ", "from "))]
        assert not any("frontier" in ln for ln in import_lines), \
            f"generated code must not import frontier: {import_lines}"
        # provenance carries the spec for champion recovery + a backend tag
        assert prog.provenance["neural_spec"]["modality"] == spec.modality
        assert prog.provenance["backend_default"] in ("torch", "sklearn-standin")
    print("[ok] render compiles, defines build_estimator, self-contained, tagged backend")


def test_build_estimator_returns_standin_without_torch():
    if _has_torch():
        pytest.skip("torch present: build_estimator returns the torch net here")
    prog = render_program(_tabular_clf_spec())
    ns = {}
    exec(compile(prog.code, "<candidate>", "exec"), ns)
    est = ns["build_estimator"]()
    assert est.backend_ == "sklearn-standin"
    assert host_backend() == "sklearn-standin"
    print("[ok] no-torch host: build_estimator -> SklearnMLPStandin, backend tag honest")


# ----------------------------------------------------------- end-to-end through the FROZEN gate

def _bc_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.80, name="bc")


def test_standin_certifies_end_to_end_through_sealed_gate():
    """A NeuralSpec -> Program runs in the REAL subprocess sandbox, is scored on val, and the
    winner is certified ONCE on the sealed test through the unmodified frozen gate. No torch.

    This is the design-03 acceptance criterion (a): the full certify loop for a neural-authored
    candidate, demonstrable today.
    """
    task = _bc_task()
    spec = NeuralSpec(modality="tabular", task_kind="classification",
                      in_features=task.n_features, out_dim=len(task.labels),
                      blocks=[Block(kind="mlp", width=64, activation="relu", norm="batch"),
                              Block(kind="residual_mlp", width=64, residual=True)],
                      lr=1e-3, max_epochs=300, seed=0)
    assert validate(spec) == []
    prog = render_program(spec)

    splits = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)

    res = sandbox.run_program(prog, Xtr, ytr, Xva, kind=task.kind, wall_seconds=90, cpu_seconds=85)
    assert res.ok, f"stand-in must run in the sandbox: [{res.error_kind}] {res.error}"
    assert res.preds is not None and len(res.preds) == len(splits.val_rows)
    val_score = certify.score_val(task, splits.val_rows, res.preds)
    assert 0.0 <= val_score <= 1.0

    # certify the winner on the sealed test (the only time sealed is touched)
    Xse = Task.rows_to_X(splits.sealed_rows)
    final = sandbox.run_program(prog, Xtr, ytr, Xse, kind=task.kind, wall_seconds=90, cpu_seconds=85)
    assert final.ok, f"winner must re-fit on sealed: [{final.error_kind}] {final.error}"
    cert = certify.certify_on_sealed(task, splits, final.preds)
    assert cert["certified"] in (True, False)
    assert cert["peeks"] == 1, "exactly one sealed peek"
    assert cert["lower_bound"] <= cert["observed"] + 1e-9
    print(f"[ok] stand-in e2e: val={val_score:.4f} sealed_lb={cert['lower_bound']} "
          f"theta={cert['theta']} certified={cert['certified']} peeks={cert['peeks']}")


def test_standin_certifies_via_engine_proposer_path():
    """Drive the FROZEN engine with NASProposer as the only source; assert it yields a neural
    winner certified through the same sealed gate. Confirms the proposer-list wiring (design §6).
    """
    if _has_torch():
        pytest.skip("torch present: engine would run the torch net (covered by pod tests)")
    from frontier.engine import ResearchEngine, EngineConfig
    task = _bc_task()
    nas = NASProposer(modality="tabular", param_budget=2_000_000, seed=0, max_proposals=3)
    res = ResearchEngine(EngineConfig(rounds=2, wall_seconds=90, cpu_seconds=85),
                         proposers=[nas]).run(task)
    assert res.winner is not None, "engine should find a runnable neural winner"
    assert res.winner.source in ("nas", "llm")
    assert res.certificate is not None and res.certificate["peeks"] == 1
    assert "neural_spec" in res.winner.provenance
    print(f"[ok] engine+NAS: winner={res.winner.label} certified={res.certified} "
          f"backend={res.winner.provenance.get('backend_default')}")


# ----------------------------------------------------------------------- proposer protocol

def _ctx(**over):
    base = {"task_kind": "classification", "n_features": 30, "n_train": 300, "round": 0,
            "tried_labels": set(), "best_label": None, "best_score": None, "best_id": None,
            "best_recipe": None, "recent_errors": [], "labels": ["0", "1"]}
    base.update(over)
    return base


def test_nas_seed_sampling_valid_in_budget_deduped():
    nas = NASProposer(modality="tabular", param_budget=2_000_000, seed=1, max_proposals=5)
    progs = nas.propose(_ctx())
    assert progs, "NAS must emit seed proposals with no champion"
    assert len(progs) <= 5
    labels = [p.label for p in progs]
    assert len(labels) == len(set(labels)), "proposals deduped by fingerprint"
    for p in progs:
        spec = NeuralSpec.from_dict(p.provenance["neural_spec"])
        assert validate(spec) == []
        assert estimate_params(spec) <= 2_000_000
    print(f"[ok] NAS seeds: {len(progs)} valid, in-budget, deduped")


def test_nas_budget_rejects_bloated_specs():
    """A tiny budget must yield zero proposals (free rejection before any build, innovation 2)."""
    nas = NASProposer(modality="tabular", param_budget=10, seed=2, max_proposals=5)
    progs = nas.propose(_ctx())
    assert progs == [], "no spec fits a 10-param budget"
    print("[ok] NAS rejects all over-budget specs for free")


def test_nas_diagnosis_conditioned_mutation_branches():
    """underfit -> widen/deepen; overfit -> dropout/norm; plateau -> add_block (design §4.1)."""
    parent = _tabular_clf_spec()
    parent_dict = parent.to_dict()
    nas = NASProposer(modality="tabular", param_budget=5_000_000, seed=0, max_proposals=10)

    def mutations(diag):
        ctx = _ctx(best_spec=parent_dict, diagnosis=diag, param_budget=5_000_000)
        progs = nas.propose(ctx)
        return [p.provenance["neural_spec"]["provenance"].get("mutation") for p in progs]

    under = mutations({"underfit": True})
    assert "widen" in under and "deepen" in under, f"underfit branch: {under}"

    over = mutations({"overfit": True})
    assert ("add_dropout" in over or "add_norm" in over or "narrow" in over
            or "raise_weight_decay" in over), f"overfit branch: {over}"

    plat = mutations({"plateau": True})
    assert "add_block" in plat or "switch_block_kind" in plat, f"plateau branch: {plat}"
    print(f"[ok] diagnosis-conditioned mutation: under={under[:3]} over={over[:3]} plat={plat[:3]}")


def test_nas_recovers_champion_from_best_recipe_fallback():
    """Even if the engine surfaces only best_recipe (not best_spec), NAS recovers the parent
    spec via best_recipe.neural_spec (the render layer mirrors it there)."""
    parent = _tabular_clf_spec()
    prog = render_program(parent)
    ctx = _ctx(best_recipe=prog.provenance["recipe"], diagnosis={"underfit": True},
               param_budget=5_000_000)
    nas = NASProposer(modality="tabular", param_budget=5_000_000)
    progs = nas.propose(ctx)
    assert progs and all(p.provenance["neural_spec"]["provenance"].get("parent")
                         == parent.fingerprint for p in progs)
    print("[ok] NAS recovers champion spec from best_recipe fallback")


def test_neural_fit_signal_underfit_overfit():
    assert neural_fit_signal(0.5, 0.5)["underfit"] is True
    assert neural_fit_signal(0.5, 0.5)["overfit"] is False
    assert neural_fit_signal(0.99, 0.7)["overfit"] is True
    assert neural_fit_signal(0.95, 0.93)["overfit"] is False
    assert neural_fit_signal(None, 0.5) == {"underfit": False, "overfit": False}
    print("[ok] neural_fit_signal distinguishes underfit/overfit, degrades on missing scores")


def test_llm_architect_inactive_without_client():
    p = LLMArchitectProposer(client=None)
    assert p.propose(_ctx()) == [], "no client -> no fabrication, honest empty"
    print("[ok] LLMArchitectProposer honestly inactive with no client")


def test_llm_architect_parses_validates_renders():
    """A stub client returns a valid spec JSON; the proposer parses, validates, renders."""
    spec_json = (
        '{"modality":"tabular","task_kind":"classification","in_features":30,"out_dim":2,'
        '"blocks":[{"kind":"mlp","width":48,"activation":"gelu","norm":"batch"},'
        '{"kind":"residual_mlp","width":48,"residual":true}],'
        '"optimizer":"adamw","lr":0.001,"weight_decay":0.01,"batch_size":128,'
        '"max_epochs":150,"patience":15,"seed":7}'
    )

    def client(_prompt):
        return "Here is the spec:\n" + spec_json + "\nthanks"

    p = LLMArchitectProposer(client=client, modality="tabular", param_budget=2_000_000, n=1)
    progs = p.propose(_ctx())
    assert len(progs) == 1 and progs[0].source == "llm"
    spec = NeuralSpec.from_dict(progs[0].provenance["neural_spec"])
    assert validate(spec) == [] and spec.provenance["author"] == "llm"
    print("[ok] LLMArchitectProposer parses JSON, validates, renders a Program")


def test_llm_architect_rejects_invalid_spec_records_error():
    """An invalid spec (regression with out_dim 3) is rejected and the violation recorded."""
    bad = ('{"modality":"tabular","task_kind":"regression","in_features":8,"out_dim":3,'
           '"blocks":[{"kind":"mlp","width":16}],"lr":0.001}')

    def client(_p):
        return bad

    p = LLMArchitectProposer(client=client, n=1)
    progs = p.propose(_ctx(task_kind="regression"))
    assert progs == [], "invalid spec must not be emitted"
    assert any("invalid spec" in e for e in p.last_parse_errors)
    print(f"[ok] LLMArchitect rejects invalid spec, records: {p.last_parse_errors[0][:60]}")


def test_llm_architect_handles_unparseable_output():
    def client(_p):
        return "I cannot help with that."

    p = LLMArchitectProposer(client=client, n=1)
    assert p.propose(_ctx()) == []
    assert any("could not parse" in e for e in p.last_parse_errors)
    print("[ok] LLMArchitect records a parse failure on non-JSON output")


# ----------------------------------------------------------------------- torch path (gated)

@pytest.mark.skipif(not _has_torch(), reason="torch only on pod")
def test_torch_net_trains_and_predicts_shape():
    spec = NeuralSpec(modality="tabular", task_kind="classification", in_features=10, out_dim=2,
                      blocks=[Block(kind="mlp", width=32, norm="batch"),
                              Block(kind="residual_mlp", width=32, residual=True)],
                      max_epochs=8, patience=4, seed=0)
    prog = render_program(spec)
    ns = {}
    exec(compile(prog.code, "<c>", "exec"), ns)
    est = ns["build_estimator"]()
    assert est.backend_ == "torch"
    rng = np.random.RandomState(0)
    X = rng.randn(120, 10).astype(np.float32)
    y = np.array(["0" if v < 0 else "1" for v in X[:, 0]], dtype=object)
    est.fit(X, y)
    preds = est.predict(X[:20])
    assert len(preds) == 20 and set(map(str, preds)) <= {"0", "1"}
    print("[ok] torch net trains + predicts right shape/labels")


@pytest.mark.skipif(not _has_torch(), reason="torch only on pod")
def test_torch_net_deterministic_same_seed():
    spec = NeuralSpec(modality="tabular", task_kind="classification", in_features=8, out_dim=2,
                      blocks=[Block(kind="mlp", width=16)], max_epochs=6, seed=123)
    prog = render_program(spec)
    rng = np.random.RandomState(1)
    X = rng.randn(80, 8).astype(np.float32)
    y = np.array(["0" if v < 0 else "1" for v in X[:, 1]], dtype=object)

    def run():
        ns = {}
        exec(compile(prog.code, "<c>", "exec"), ns)
        est = ns["build_estimator"]()
        est.fit(X, y)
        return list(map(str, est.predict(X[:30])))

    assert run() == run(), "same seed must yield identical predictions"
    print("[ok] torch net deterministic under fixed seed")


@pytest.mark.skipif(not _has_torch(), reason="torch only on pod")
def test_torch_net_certifies_through_sealed_gate():
    task = _bc_task()
    spec = NeuralSpec(modality="tabular", task_kind="classification",
                      in_features=task.n_features, out_dim=len(task.labels),
                      blocks=[Block(kind="mlp", width=64, norm="batch"),
                              Block(kind="residual_mlp", width=64, residual=True)],
                      max_epochs=60, patience=10, seed=0)
    prog = render_program(spec)
    splits = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xse = Task.rows_to_X(splits.sealed_rows)
    final = sandbox.run_program(prog, Xtr, ytr, Xse, kind=task.kind, wall_seconds=300, cpu_seconds=290)
    assert final.ok, f"[{final.error_kind}] {final.error}"
    cert = certify.certify_on_sealed(task, splits, final.preds)
    assert cert["peeks"] == 1
    print(f"[ok] torch net certifies through sealed gate: certified={cert['certified']}")


def _run_all():
    import warnings
    warnings.simplefilter("ignore")
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for name, fn in fns:
        mark = getattr(fn, "pytestmark", [])
        skip = any(getattr(m, "name", "") == "skipif" and m.args and m.args[0] for m in mark)
        if skip and not _has_torch():
            skipped += 1
            print(f"[skip] {name}: torch only on pod")
            continue
        try:
            fn()
        except pytest.skip.Exception as e:  # type: ignore[attr-defined]
            skipped += 1
            print(f"[skip] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} passed, {skipped} skipped, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
