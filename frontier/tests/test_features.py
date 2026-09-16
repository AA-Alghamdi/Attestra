"""Phase-1 feature-engineering proposal-source tests.

Run standalone:
    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_features.py

Asserts BEHAVIOR (not just import):
  - the recipe genome normalizes idempotently and obeys the safe-composition rules;
  - every emitted code string compiles, defines build_estimator(), and the estimator
    actually fits+predicts on a REAL low-dim dataset (diabetes, 10 features);
  - FeatureProposer proposes feature-engineered + target-transform recipes for a 10-feature
    regression task (the n_features>=60 gate is dead);
  - FeatureMutationProposer is a BOUNDED beam, uses recent_errors to deprioritize, and does
    crossover toward archetypes;
  - end to end through the Phase-0 engine on diabetes, a feature-engineered recipe certifies
    ABOVE the plain-model (ridge) floor on the held-out sealed test (one counted peek).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, sandbox
from frontier.engine import ResearchEngine, EngineConfig
from frontier.program import Program
from frontier.proposers import SeedProposer, MutationProposer
from frontier.task import Task
from frontier.features import (
    FeatureProposer, FeatureMutationProposer, make_code, normalize_genome,
    recipe_label, _archetypes,
)


def _diabetes_task(theta: float = 0.40) -> Task:
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return Task(X=d.data, y=d.target.astype(float), kind="regression",
                theta=theta, metric="r2", name="diabetes")


def _diabetes_arrays():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return d.data.astype(float), d.target.astype(float)


def test_normalize_is_idempotent_and_arbitrates():
    # rule 2: power + standard scale -> standard scale dropped.
    g = normalize_genome({"base": "ridge", "power": "yeo-johnson", "scale": "standard"}, "regression")
    assert g["power"] == "yeo-johnson" and g["scale"] is False, g
    # rule 3: reduce + select -> select dropped.
    g2 = normalize_genome({"base": "ridge", "reduce": ("pca", 0.95), "select": ("kbest_mi", 0.5)},
                          "regression")
    assert g2["reduce"] is not None and g2["select"] is None, g2
    # rule 5: target transform dropped for classification.
    g3 = normalize_genome({"base": "logreg", "target": "log1p"}, "classification")
    assert g3["target"] is None, g3
    # idempotence.
    assert normalize_genome(g, "regression") == g
    assert normalize_genome(g2, "regression") == g2
    print(f"[ok] genome normalization + safe-composition rules: {recipe_label(g)}, {recipe_label(g2)}")


def test_all_archetype_code_compiles_and_fits():
    X, y = _diabetes_arrays()
    Xtr, ytr, Xev = X[:300], y[:300], X[300:]
    n_ok = 0
    for g in _archetypes("regression"):
        code = make_code(g, "regression")
        assert "def build_estimator" in code
        ns: dict = {}
        exec(compile(code, "<g>", "exec"), ns)            # must compile
        est = ns["build_estimator"]()                     # must build
        est.fit(Xtr, ytr)                                 # must fit on real data
        preds = est.predict(Xev)                          # must predict
        assert len(preds) == len(Xev) and np.all(np.isfinite(np.asarray(preds, float)))
        n_ok += 1
    assert n_ok == len(_archetypes("regression"))
    print(f"[ok] {n_ok}/{n_ok} archetype pipelines compile + fit + predict on diabetes")


def test_runtime_size_clamps_for_poly_blowup():
    # poly2 on 10 features -> 55 features; a 0.5 k-best must resolve to <= 55 at fit time, not crash.
    X, y = _diabetes_arrays()
    g = normalize_genome({"base": "ridge", "scale": "standard", "poly": 2,
                          "select": ("kbest_mi", 0.5)}, "regression")
    code = make_code(g, "regression")
    ns: dict = {}
    exec(compile(code, "<g>", "exec"), ns)
    est = ns["build_estimator"]()
    est.fit(X[:300], y[:300])
    assert est.predict(X[300:]).shape[0] == len(X) - 300
    print(f"[ok] runtime-resolved selection survives poly blowup: {recipe_label(g)}")


def test_feature_proposer_kills_lowdim_gate():
    # 10 features -> still proposes feature engineering + a target transform.
    props = FeatureProposer().propose({"task_kind": "regression", "tried_labels": set()})
    labels = {p.label for p in props}
    assert props, "FeatureProposer must propose for low-dim data"
    assert any("poly2" in l for l in labels), f"expected poly interaction recipe: {labels}"
    assert any(l.startswith(("tlog", "tyj", "tqn")) for l in labels), f"expected target transform: {labels}"
    assert any("yj" in l or "qn" in l for l in labels), f"expected a power transform: {labels}"
    # every proposal carries a genome in provenance the engine can read back.
    assert all("recipe" in p.provenance for p in props)
    print(f"[ok] low-dim (10-feat) feature engineering proposable: {len(labels)} recipes")


def test_mutation_is_bounded_and_error_aware():
    champ = normalize_genome({"base": "ridge", "scale": "standard"}, "regression")
    fmp = FeatureMutationProposer(beam_width=8)
    base_ctx = {"task_kind": "regression", "tried_labels": set(),
                "best_recipe": champ, "best_id": "champ"}
    muts = fmp.propose(base_ctx)
    assert 0 < len(muts) <= 8, f"beam must be bounded by beam_width: got {len(muts)}"
    assert all(m.label != recipe_label(champ) for m in muts), "must not re-emit the champion"
    # determinism: same context -> same proposals.
    muts2 = fmp.propose(base_ctx)
    assert [m.label for m in muts] == [m.label for m in muts2], "beam must be deterministic"

    # error-awareness: marking 'poly2' tokens as failing should push poly2 recipes down the beam.
    poly_label = next((m.label for m in muts if "poly2" in m.label), None)
    if poly_label is not None:
        err_ctx = dict(base_ctx)
        err_ctx["recent_errors"] = [(poly_label, "fit", "boom")] * 3
        muts_err = fmp.propose(err_ctx)
        rank_no_err = [m.label for m in muts].index(poly_label) if poly_label in [m.label for m in muts] else 999
        labels_err = [m.label for m in muts_err]
        rank_err = labels_err.index(poly_label) if poly_label in labels_err else 999
        assert rank_err >= rank_no_err, "failing token must not be promoted in ordering"
    print(f"[ok] bounded beam (<=8), deterministic, error-aware: {[m.label for m in muts]}")


def test_crossover_seeds_from_archetypes_with_no_champion():
    # with no champion yet, the proposer still works (seeds genome from an archetype).
    muts = FeatureMutationProposer(beam_width=6).propose(
        {"task_kind": "regression", "tried_labels": set(), "best_recipe": None})
    assert 0 < len(muts) <= 6
    print(f"[ok] cold-start (no champion) still proposes: {[m.label for m in muts][:6]}")


def test_emitted_program_runs_in_real_sandbox():
    # the firewall path: emit code, run it out-of-process, get predictions only.
    task = _diabetes_task()
    s = certify.make_splits(task, seed=0)
    g = normalize_genome({"base": "ridge", "scale": "standard", "poly": 2,
                          "interactions_only": True}, "regression")
    prog = Program(code=make_code(g, "regression"), source="feature", label=recipe_label(g))
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xva = Task.rows_to_X(s.val_rows)
    res = sandbox.run_program(prog, Xtr, ytr, Xva, kind="regression", wall_seconds=60)
    assert res.ok, f"feature-engineered program should run: [{res.error_kind}] {res.error}"
    assert res.preds is not None and len(res.preds) == len(s.val_rows)
    assert not hasattr(res, "score")
    print(f"[ok] emitted FE program ran in sandbox: {len(res.preds)} preds, [{recipe_label(g)}]")


def test_end_to_end_feature_eng_beats_plain_floor_and_certifies():
    """The headline acceptance: a feature-engineered recipe certifies above the plain-model
    floor on the held-out sealed test, on a 10-feature regression task."""
    task = _diabetes_task(theta=0.40)

    # (a) plain-model floor: SeedProposer + MutationProposer only (no feature module).
    plain = ResearchEngine(EngineConfig(rounds=2, seed=0, wall_seconds=60, cpu_seconds=55),
                           proposers=[SeedProposer(), MutationProposer()]).run(task)
    assert plain.winner is not None and plain.certificate is not None
    floor_obs = plain.certificate["observed"]

    # (b) with the feature-engineering sources added.
    fe = ResearchEngine(
        EngineConfig(rounds=3, seed=0, wall_seconds=60, cpu_seconds=55),
        proposers=[SeedProposer(), FeatureProposer(), MutationProposer(),
                   FeatureMutationProposer(beam_width=8)],
    ).run(task)
    assert fe.winner is not None, "FE run must find a winner"
    assert fe.certificate is not None and fe.certificate["peeks"] == 1, "one sealed peek"
    c = fe.certificate
    assert c["lower_bound"] <= c["observed"] + 1e-9, "certified bound is the LOWER bound"
    assert c["certified"], f"FE recipe must clear theta on sealed: {c}"

    # the winner must actually be a feature-engineered or feature-mutated program (the point).
    assert fe.winner.source in ("feature", "feature_mut"), \
        f"winner should come from the feature module, got source={fe.winner.source} label={fe.winner.label}"
    # and it must beat the plain floor's observed sealed score.
    assert c["observed"] >= floor_obs - 1e-9, \
        f"FE winner sealed observed {c['observed']} should be >= plain floor {floor_obs}"

    print(f"[ok] e2e: plain floor sealed obs={floor_obs:.4f}; "
          f"FE winner={fe.winner.label} ({fe.winner.source}) "
          f"val={fe.winner_val_score} sealed obs={c['observed']:.4f} lb={c['lower_bound']:.4f} "
          f"theta={c['theta']} certified={c['certified']}")


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
    print(f"\n{len(fns) - failed}/{len(fns)} feature-phase tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
