"""Hermetic locks for OPEN-ENDED AUTHORED-CODE discovery under sealed certification.

These prove the code axis is genuinely load-bearing AND that the frozen governance still holds over it:
  1. SANDBOX ADMITS SAFE: a template-authored featurizer + classifier clear the frozen three-stage gate and
     EXECUTE on real features (run_authored_predict returns predictions).
  2. SANDBOX REJECTS UNSAFE: source that does `import os` is rejected at the static AST gate -- before any
     execution -- so it can never reach a sealed peek.
  3. CASCADE REJECTS NON-IMPROVING CODE: on a LINEARLY-SEPARABLE problem (linear baseline already ~optimal),
     authored code IS proposed every features round but NONE is certified -> the champion keeps no code
     (menu_free stays False). The certifier, not optimism, decides.
  4. GENUINE IMPROVEMENT CERTIFIES: on a NONLINEAR problem (linear ~chance), an authored classifier the
     system wrote is frozen-Tier-3 certified over the linear champion, gold-confirmed on never-peeked rows
     -> menu_free=True, uses_authored_code=True. The winning SOURCE was never handed to the system.

Everything is OFFLINE (TemplateAuthorer; no LLM call) and runs through the REAL frozen primitives
(authoring's three-stage admit + battery.mcnemar/bh + science.clopper_pearson_lower) -- nothing is stubbed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from code_arena import TabularCodeArena, TabularSplits  # noqa: E402

from vfplatform import authoring_bridge as AB  # noqa: E402
from vfplatform.code_authoring import TemplateAuthorer  # noqa: E402
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator  # noqa: E402
from vfplatform.recipe_research import RecipeResearcher  # noqa: E402


# ---------------------------------------------------------------------------- synthetic datasets (offline)
def _nonlinear_multiclass(n=900, seed=0):
    """A 3-class problem with NO linear separation: class = quadrant-style sign interactions. A linear head
    is ~chance; a tree/kernel classifier recovers it. Real structure, tiny + fast."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = ((X[:, 0] * X[:, 1] > 0).astype(int) + (X[:, 2] * X[:, 3] > 0).astype(int)).astype(int)  # 0,1,2
    return X, y, 3, "synthetic_nonlinear"


def _separable_multiclass(n=900, seed=0):
    """Well-separated Gaussian blobs: a linear head is ~optimal, so authored code has no headroom to win."""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=3, n_features=6, cluster_std=0.6, random_state=seed)
    return X.astype(float), y.astype(int), 3, "synthetic_separable"


def _nonlinear_regression(n=1500, seed=0):
    """A CONTINUOUS target with squares + interactions + a sinusoid: a linear model lands BELOW the within-tau
    bar (~0.43) and a constant (median) predictor is below it too (~0.47 < theta=0.5), while a tree/kernel
    regressor clears it decisively (~0.78-0.89). n_classes==0 marks regression."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.sin(3 * X[:, 3]) + 0.1 * rng.normal(size=n)).astype(float)
    return X, y, 0, "synthetic_nonlinear_reg"


def _grouped_invariant_regression(n_groups=18, per=110, shift_sd=1.0, seed=0):
    """A GROUP-structured regression whose signal is GROUP-INVARIANT: every group shares the same nonlinear
    target y=x0^2 + x1*x2 + 0.7*sin(3*x3) + noise; groups differ only by a modest covariate shift (so a
    held-out group INTERPOLATES the others). A linear model is ~0.4 within-tau, a tree/kernel regressor ~0.83
    -- and crucially the tree's edge GENERALIZES to groups it never trained on. n_classes==0 (regression)."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for gid in range(n_groups):
        c = rng.normal(0, shift_sd, size=5)
        Xg = rng.normal(c, 1.0, size=(per, 5))
        yg = (Xg[:, 0] ** 2 + Xg[:, 1] * Xg[:, 2] + 0.7 * np.sin(3 * Xg[:, 3]) + rng.normal(0, 0.3, size=per))
        X.append(Xg); y.append(yg); g.append(np.full(per, gid))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def _arena(data, *, shape="multiclass", seed=0, n_shards=3):
    splits = TabularSplits(dataset="synthetic", shape=shape, n_shards=n_shards, seed=seed, data=data)
    return TabularCodeArena(dataset="synthetic", shape=shape, splits=splits)


def _researcher(arena, *, seed=0, peeks=10, roles=("featurizer", "classifier")):
    cfg = GeneratorConfig(task_hint=arena.task_hint, fanout=6, code_prob=1.0,
                          code_roles=tuple(roles), graft_prob=0.0, online_discovery=False)
    gen = RecipeGenerator(arena.seed_recipes(), config=cfg, seed=seed,
                          code_authorer=TemplateAuthorer(seed=seed), discover_fn=(lambda: []),
                          task_shape=arena.task_shape)
    return RecipeResearcher(arena, gen, alpha=0.1, theta_floor=arena.theta_floor, peek_budget=peeks,
                            competence_ceiling=0.999, allow_data_acquisition=False), gen


# ---------------------------------------------------------------------------- 1. sandbox admits safe
def test_sandbox_admits_and_executes_safe_authored_code():
    X, y, k, _ = _nonlinear_multiclass()
    tr, ev = np.arange(700), np.arange(700, 900)
    ta = TemplateAuthorer(seed=1)
    for role in ("featurizer", "classifier"):
        src = ta(role, "multiclass")
        pred, report = AB.run_authored_predict(src, role, X, y, tr, ev, n_classes=k, family="raw", seed=0)
        assert report.admitted, f"{role} should pass the frozen sandbox: {report.reason}"
        assert pred is not None and len(pred) == len(ev)


# ---------------------------------------------------------------------------- 2. sandbox rejects unsafe
def test_sandbox_rejects_unsafe_authored_code():
    X, y, k, _ = _nonlinear_multiclass()
    tr, ev = np.arange(700), np.arange(700, 900)
    unsafe = "import os\ndef build_estimator(seed):\n    os.system('echo pwned')\n    return None\n"
    pred, report = AB.run_authored_predict(unsafe, "classifier", X, y, tr, ev, n_classes=k)
    assert pred is None and not report.admitted
    assert "os" in report.reason and "not allowed" in report.reason

    net = ("import numpy as np\nimport socket\n"
           "def build_estimator(seed):\n    return np\n")
    pred2, rep2 = AB.run_authored_predict(net, "featurizer", X, y, tr, ev, n_classes=k)
    assert pred2 is None and not rep2.admitted


# ---------------------------------------------------------------------------- 3. cascade rejects non-improving
def test_cascade_rejects_non_improving_authored_code():
    arena = _arena(_separable_multiclass())
    researcher, gen = _researcher(arena, peeks=10)

    # authored code IS on the table every features round (prove it was proposed, not silently skipped)
    children = gen.expand(arena.seed_recipes()[0], "features")
    assert any(c.code_patch for c in children), "generator must propose code-bearing recipes"

    cert = researcher.run()
    # linear is already ~optimal -> no authored patch earns a certified positive lift -> none promoted
    assert not cert.novelty["uses_authored_code"]
    assert not cert.refused
    assert all(p.rung != "features" or "code" not in p.to_recipe for p in cert.promotions) or \
        cert.champion_recipe.get("code_patch") is None


# ---------------------------------------------------------------------------- 4. genuine improvement certifies
def test_authored_classifier_certifies_on_genuine_improvement():
    arena = _arena(_nonlinear_multiclass())
    researcher, gen = _researcher(arena, peeks=12)
    cert = researcher.run()

    assert not cert.refused, "clean framing should not refuse"
    assert cert.novelty["menu_free"] is True
    assert cert.novelty["uses_authored_code"] is True
    assert cert.novelty["backbone_is_novel"] is False         # the representation was PINNED to raw features
    assert cert.champion_recipe.get("code_patch")             # the certificate carries the authored SOURCE
    assert cert.champion_recipe["code_role"] in ("featurizer", "classifier")

    # the win is real on the sealed rows (clearly above 3-class chance) and survives the gold read
    assert min(cert.sealed_acc.values()) > 0.55          # 3-class chance is ~0.33; linear champ is ~0.43
    assert cert.gold_confirmation is not None and cert.gold_confirmation["confirmed"] is True

    # at least one promotion came from the features (code) rung with a SUBSTANTIAL positive lift
    code_promos = [p for p in cert.promotions if p.rung == "features" and p.mean_lift > 0]
    assert code_promos, "a genuinely-better authored patch must be certified over the linear champion"
    assert max(p.mean_lift for p in code_promos) > 0.05, "the certified authored lift must be material"


# ------------------------------------------------------ 4b. regression certifies via tolerance-Bernoulli
def test_regression_tolerance_bernoulli_certifies():
    """REGRESSION task-shape under the IDENTICAL frozen certifier. Correctness is the per-row Bernoulli
    hit |pred-y|<=tau, so paired McNemar + BH-FDR + Clopper-Pearson certify regression BYTE-IDENTICALLY (no
    new statistical primitive). On a nonlinear target the linear seed is below the within-tau bar and a
    nonlinear lever certifies a MATERIAL lift, gold-confirmed -- and an AUTHORED regressor (source the system
    wrote, admitted through the frozen sandbox) is itself frozen-certified over the linear seed."""
    arena = _arena(_nonlinear_regression(), shape="regression")
    assert arena.is_regression and arena.task_shape == "regression" and arena.tau > 0

    # the tolerance metric is NOT gameable by predicting the center: a constant (median) predictor is below
    # theta on the sealed rows -> the run cannot be won by ignoring the features.
    sp = arena.splits
    sealed = np.concatenate(sp.shards)
    med = float(np.median(sp.y[sp.train_idx]))
    median_hit = float(arena._hit(np.full(len(sealed), med), sp.y[sealed]).mean())
    assert median_hit < arena.theta_floor, f"trivial median predictor must fail theta (got {median_hit:.3f})"

    researcher, _ = _researcher(arena, peeks=12, roles=("regressor",))
    cert = researcher.run()

    assert not cert.refused, "clean regression framing/data-hygiene should NOT refuse"
    # the nonlinear lever fires: champion clears the bar with a material certified lift, gold-confirmed.
    assert min(cert.sealed_acc.values()) > 0.60, cert.sealed_acc        # linear seed is ~0.43
    promos = [p for p in cert.promotions if p.mean_lift > 0]
    assert promos and max(p.mean_lift for p in promos) > 0.10, "a material lift over linear must certify"
    assert cert.gold_confirmation is not None and cert.gold_confirmation["confirmed"] is True

    # an AUTHORED regressor (source the system wrote) was frozen-Tier-3 CERTIFIED over the linear seed under
    # the tolerance-Bernoulli metric -- the code axis is load-bearing on regression too, not just the menu.
    assert any(("code[regressor]" in ln and "CERTIFIED" in ln) for ln in cert.log), \
        "an authored regressor must reach a frozen certification on a nonlinear regression target"


# ----------------------------------------------- 4c. regression data-hygiene admits a continuous target
def test_regression_data_hygiene_admits_continuous_target():
    """The kNN label-DISAGREEMENT + class-balance hygiene checks are classification-only; on a continuous
    target the discrete-equality disagreement is ~1.0 by construction and would FALSELY refuse a clean
    dataset. is_regression=True skips them (the non-negotiable near-duplicate-straddle leak still blocks),
    so a clean regression split is ADMITTED."""
    from vfplatform import data_cert as DC

    X, y, _, _ = _nonlinear_regression(n=600)
    tr, se = np.arange(400), np.arange(400, 600)
    # the classification path mis-reads a continuous target as almost-all-noise and refuses ...
    rep_cls = DC.certify_dataset(X, y, train_idx=tr, sealed_idx=se, is_regression=False)
    assert rep_cls.label_noise_est is not None and rep_cls.label_noise_est > 0.5 and not rep_cls.passed
    # ... the regression path skips the inapplicable check and ADMITS the clean split.
    rep_reg = DC.certify_dataset(X, y, train_idx=tr, sealed_idx=se, is_regression=True)
    assert rep_reg.label_noise_est is None and rep_reg.near_dup_straddle == 0 and rep_reg.passed


# ----------------------------------------- 4d. GROUPED split: certify a lever that generalizes + block leak
def test_grouped_split_certifies_generalizing_lever_and_refuses_leak():
    """GROUPED task-shape: whole groups are held out, so the sealed + gold rows belong to groups the model
    NEVER trained on (a covariate-shift / extrapolation test, not i.i.d.). Two locks:
      (a) the carve leaks no group, and on a GROUP-INVARIANT nonlinear signal the loop certifies a material
          lift that is GOLD-confirmed on never-peeked HELD-OUT groups (the lever genuinely generalizes); an
          authored regressor is itself frozen-certified under the grouped split.
      (b) a deliberately leaky grouped framing (a sealed row sharing a train group) makes the meta-certifier
          REFUSE the whole run before a single sealed peek is spent."""
    X, y, groups = _grouped_invariant_regression()
    sp = TabularSplits(dataset="synthetic", shape="regression", split="grouped", groups=groups,
                       data=(X, y, 0, "synthetic_grouped_reg"), n_total=len(y), seed=0)
    arena = TabularCodeArena(dataset="synthetic", shape="regression", splits=sp)

    # the carve holds out WHOLE groups: no group straddles train / sealed / gold
    g_train = set(groups[sp.train_idx])
    g_sealed = set().union(*[set(groups[s].tolist()) for s in sp.shards])
    g_gold = set(groups[sp.gold_idx])
    assert g_train.isdisjoint(g_sealed) and g_train.isdisjoint(g_gold), "a group straddled a split boundary"
    assert arena.framing().get("groups") is not None, "framing must re-expose groups for the leak probe"

    researcher, _ = _researcher(arena, peeks=12, roles=("regressor",))
    cert = researcher.run()
    assert not cert.refused, "a clean grouped split must not refuse"
    assert cert.framing_report and cert.framing_report["trustworthy"] is True
    promos = [p for p in cert.promotions if p.mean_lift > 0]
    assert promos and max(p.mean_lift for p in promos) > 0.10, "a generalizing lever must certify a material lift"
    # gold confirmation is on HELD-OUT groups -> proves generalization across groups, not memorization
    assert cert.gold_confirmation is not None and cert.gold_confirmation["confirmed"] is True
    assert any(("code[regressor]" in ln and "CERTIFIED" in ln) for ln in cert.log), \
        "an authored regressor must reach a frozen certification under the grouped split"

    # (b) a leaky grouped split is REFUSED end-to-end (a sealed row is forced to share a train group id)
    class _LeakyGroupArena(TabularCodeArena):
        def framing(self):
            fr = super().framing()
            g = np.asarray(fr["groups"]).copy()
            g[len(fr["train_idx"])] = g[0]                 # a sealed row now shares a train group -> leak
            fr["groups"] = g
            return fr

    leaky = _LeakyGroupArena(dataset="synthetic", shape="regression", splits=sp)
    researcher2, _ = _researcher(leaky, peeks=12, roles=("regressor",))
    cert2 = researcher2.run()
    assert cert2.refused, "a grouped split with a straddling group MUST be refused"
    assert cert2.peeks_used == 0, "no sealed peek may be spent on a leaky split"


# -------------------------------------------- 4e. TIME split: forward-chaining + temporal-leak is refused
def test_time_split_forward_chaining_and_refuses_temporal_leak():
    """TIME task-shape: train is the EARLIEST block, the sealed shards + gold are strictly LATER (train on
    past, certify on future). Locks: the carve is forward-chained (max train time <= min sealed time <= gold),
    a clean split is trustworthy, and a temporal leak (a training row later than a sealed row) is REFUSED."""
    X, y, _ = _grouped_invariant_regression()
    rng = np.random.default_rng(3)
    times = rng.permutation(len(y)).astype(float)          # an arbitrary (shuffled) real time index per row
    sp = TabularSplits(dataset="synthetic", shape="regression", split="time", times=times,
                       data=(X, y, 0, "synthetic_time_reg"), n_total=len(y), seed=0)
    sealed = np.concatenate(sp.shards)
    assert times[sp.train_idx].max() <= times[sealed].min(), "temporal leak: a train row is later than sealed"
    assert times[sealed].max() <= times[sp.gold_idx].min(), "gold must be the furthest-future block"

    arena = TabularCodeArena(dataset="synthetic", shape="regression", splits=sp)
    assert arena.framing().get("times") is not None, "framing must re-expose times for the temporal-leak probe"
    cert = _researcher(arena, peeks=10, roles=("regressor",))[0].run()
    assert not cert.refused and cert.framing_report["trustworthy"] is True, "a clean time split must not refuse"

    # a temporal leak (a training row stamped LATER than every sealed row) is refused end-to-end
    class _LeakyTimeArena(TabularCodeArena):
        def framing(self):
            fr = super().framing()
            t = np.asarray(fr["times"], dtype=float).copy()
            t[0] = t.max() + 1.0                           # a train row is now later than all sealed rows
            fr["times"] = t
            return fr

    leaky = _LeakyTimeArena(dataset="synthetic", shape="regression", splits=sp)
    cert2 = _researcher(leaky, peeks=10, roles=("regressor",))[0].run()
    assert cert2.refused and cert2.peeks_used == 0, "a temporal leak MUST be refused before any peek"


# ---------------------------------------------------------------------------- 5. LLM revise loop (offline)
def test_llm_authorer_feeds_prior_failures_to_revise(monkeypatch):
    """The open-ended LLMAuthorer's REVISE loop: when an authored estimator is rejected by the frozen gate,
    the NEXT attempt must carry the prior admission-failure reason in extra_context -- so the model can revise
    AND the request is cache-distinct (identical context would replay the same rejected code). Fully offline:
    authoring.author_estimator is monkeypatched; no network, no real LLM, the gate is not relaxed."""
    from vfplatform import authoring as A
    from vfplatform.code_authoring import LLMAuthorer

    seen = []

    class _Est:
        code = ("def build_estimator(seed):\n"
                "    from sklearn.ensemble import RandomForestClassifier\n"
                "    return RandomForestClassifier(n_estimators=50, random_state=seed)\n")

    def fake_author(spec, **kwargs):
        seen.append(kwargs.get("extra_context"))
        if len(seen) == 1:                                  # first attempt: rejected with a specific reason
            return A.AuthoringResult(False, None, True, "DECLINED: static gate: use of banned name 'getattr'")
        return A.AuthoringResult(True, _Est(), True, "admitted")  # second: a conforming, admitted estimator

    monkeypatch.setattr(A, "author_estimator", fake_author)
    auth = LLMAuthorer(n_features=6, n_classes=3, api_key="offline-test-key")

    first = auth("classifier", "multiclass")
    assert first is None                                    # rejected -> no source this round
    assert seen[0]["attempt"] == 1 and "prior_admission_failures" not in seen[0]

    second = auth("classifier", "multiclass")
    assert second is not None and "build_estimator" in second
    assert seen[1]["attempt"] == 2                          # the loop advanced the attempt counter
    assert "prior_admission_failures" in seen[1]            # ... and fed the rejection back to the model
    assert any("getattr" in r for r in seen[1]["prior_admission_failures"])
    # provenance for the certificate: both attempts recorded, with honest authored flags
    assert [c["attempt"] for c in auth.calls] == [1, 2]
    assert auth.calls[0]["authored"] is False and auth.calls[1]["authored"] is True


# ---------------------------------------------------------------------------- OpenML-CC18 suite plumbing
def test_openml_loader_encodes_dedups_and_caches(monkeypatch, tmp_path):
    """The CC18 loader must: one-hot categorical + median-impute numeric, DROP exact-duplicate rows (the
    hygiene fix that keeps near-dups from straddling train/sealed), label-encode the target, and cache to
    disk so the second call is offline. Fully offline here (fetch_openml is faked)."""
    import types

    import pandas as pd

    import code_arena as CA

    # a tiny frame: 1 numeric + 1 categorical column, where every (num,cat) row appears TWICE -> 4 exact dups
    df = pd.DataFrame({"num": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
                       "cat": ["a", "a", "b", "b", "a", "a", "b", "b"]})
    df["cat"] = df["cat"].astype("category")
    target = pd.Series(["x", "x", "y", "y", "x", "x", "y", "y"])

    calls = {"n": 0}

    def fake_fetch(data_id, as_frame, parser):
        calls["n"] += 1
        return types.SimpleNamespace(data=df.copy(), target=target.copy())

    monkeypatch.setattr("sklearn.datasets.fetch_openml", fake_fetch)
    real_expand = os.path.expanduser
    monkeypatch.setattr(CA.os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path), 1) if p.startswith("~") else real_expand(p))

    X, y, k, hint = CA._load_openml(990001, "synthetic")
    assert X.shape == (4, 3)          # 8 rows -> 4 after exact-dedup; 1 numeric + one-hot(2 categories) cols
    assert k == 2 and len(y) == 4 and hint == "synthetic"
    assert set(int(v) for v in y) == {0, 1}
    assert calls["n"] == 1 and os.path.exists(os.path.join(str(tmp_path), "wilds_data",
                                                            "_openml_cache", "990001.npz"))

    # second call must hit the cache, NOT the network (make the fetch explode to prove it is never called)
    def boom(*a, **k):
        raise AssertionError("fetch_openml called on a cache hit")

    monkeypatch.setattr("sklearn.datasets.fetch_openml", boom)
    X2, y2, k2, _ = CA._load_openml(990001, "synthetic")
    assert X2.shape == (4, 3) and k2 == 2 and np.array_equal(X2, X) and np.array_equal(y2, y)


def test_auto_theta_floor_is_data_driven_and_uncheatable():
    """The competence floor = max(0.5, train-majority + margin): 0.5 on balanced sets (so nothing changes),
    raised to just above the trivial baseline on imbalanced ones (so raw accuracy is not gameable). Computed
    from TRAIN labels only."""
    import types

    from run_code_discovery import _auto_theta_floor

    balanced = types.SimpleNamespace(splits=types.SimpleNamespace(
        y_train=np.array([0, 1, 2, 3] * 5), train_idx=np.arange(20)))      # 4-class, maj 0.25 -> stays 0.5
    assert _auto_theta_floor(balanced) == 0.5

    imbalanced = types.SimpleNamespace(splits=types.SimpleNamespace(
        y_train=np.array([0] * 16 + [1] * 4), train_idx=np.arange(20)))     # maj 0.80 -> 0.80 + 0.03
    assert abs(_auto_theta_floor(imbalanced) - 0.83) < 1e-9


# ---------------------------------------------------------------------------- frozen-core untouched
def test_frozen_core_hashes_unchanged():
    import hashlib

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    want = {os.path.join(root, "vectorforge", "science.py"): "b564fba2",
            os.path.join(root, "vfplatform", "sealed.py"): "30ad6245"}
    for path, prefix in want.items():
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        assert digest.startswith(prefix), f"FROZEN FILE CHANGED: {path} -> {digest[:8]} (want {prefix})"
