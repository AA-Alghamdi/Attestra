"""Hermetic locks for the REGENERATIVE autoresearcher (vfplatform/recipe_research.py).

These prove the menu-free loop is real AND that the frozen governance still holds under it:
  1. OPEN DISCOVERY: starting from a weak resnet seed, the system climbs to a NON-SEED backbone it was
     never handed -> the certificate's `menu_free` proof is True.
  2. FROZEN SOLE PROMOTER: flipping the winner's sealed labels (so it no longer beats the champion on the
     frozen paired-FDR compare) BLOCKS promotion -- promotion rides on sealed labels alone, not validation.
  3. META-CERTIFIER FRAMING GATE: a leaky framing (label-shuffle still clears theta) makes the run REFUSE
     to search -- the referee leads, optimism cannot override it.
  4. AUTHORING SANDBOX: a recipe carrying unsafe code (`import os`) is rejected before it can spend a peek.
  5. SEARCH DRIVER PARITY: the search.BudgetedSearch driver reaches a certified non-seed champion too.
  6. DATA-HYGIENE GATE: a dataset with near-duplicate train/sealed straddle (the non-negotiable leak) is
     REFUSED by the datapool gate before any sealed peek; clean data is admitted and recorded.

The synthetic arena generates per-example correctness so the comparison runs through the REAL frozen
primitives (battery.mcnemar_pvalue / benjamini_hochberg + science.clopper_pearson_lower) -- the test
exercises the certifier, it does not stub it.
"""
import random

import pytest

from vfplatform.recipe import Recipe
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator
from vfplatform.recipe_research import RecipeArena, RecipeResearcher
from vfplatform.repr_researcher import TaskMeasure

N_PER_TASK = 200
TASKS = ["t0", "t1", "t2"]

# family -> "true accuracy". dinov2 (NON-SEED) is the only family that clears the competence floor AND beats
# the resnet seed; clip is competent-but-not-winning; resnet is the weak seed. Authoring on a fixed backbone
# (adaptation/aug) moves accuracy only trivially -- so ONLY changing the backbone wins (the proven law).
_FAMILY_ACC = {"dinov2": 0.88, "clip": 0.72, "convnext": 0.70, "resnet": 0.60, "other": 0.56}


def _family(backbone: str) -> str:
    b = backbone.lower()
    for fam in ("dinov2", "clip", "convnext", "resnet"):
        if fam in b:
            return fam
    return "other"


def _acc(recipe: Recipe) -> float:
    base = _FAMILY_ACC[_family(recipe.backbone)]
    bump = 0.01 if recipe.adaptation in ("lora", "full_ft") else 0.0   # authoring barely moves it
    return min(0.97, base + bump)


def _correct(acc: float, salt: str) -> list:
    """Deterministic nested per-example correctness: example i is correct iff u_i < acc, with u_i drawn from
    a per-(task,salt) stream. Nested across recipes -> a better recipe is correct on a superset, so the
    paired McNemar sees genuine discordant pairs favoring the better recipe."""
    rng = random.Random(hash(salt) & 0xFFFFFFFF)
    return [1 if rng.random() < acc else 0 for _ in range(N_PER_TASK)]


class SyntheticRecipeArena(RecipeArena):
    tasks = TASKS
    task_hint = "fine_grained"
    task_shape = "binary"

    def __init__(self, *, leaky_framing: bool = False, flip_winner_sealed: bool = False,
                 contaminated_data: bool = False):
        self.leaky_framing = leaky_framing
        self.flip_winner_sealed = flip_winner_sealed
        self.contaminated_data = contaminated_data

    def seed_recipes(self):
        return [Recipe(backbone="resnet18", adaptation="linear_probe")]

    def measure(self, recipe: Recipe):
        acc = _acc(recipe)
        out = {}
        for t in self.tasks:
            # nested across recipes: same u-stream per task, threshold by this recipe's accuracy
            sealed = _correct(acc, f"sealed::{t}")
            val = _correct(acc, f"val::{t}")
            # adversarial knob: destroy the winner's sealed signal so the frozen compare cannot certify it,
            # while leaving validation (which the cheap screen reads) intact -> tests sole-promoter on sealed.
            if self.flip_winner_sealed and _family(recipe.backbone) == "dinov2":
                sealed = [0] * N_PER_TASK
            out[t] = TaskMeasure(sealed_correct=sealed, val_correct=val,
                                 acc=sum(sealed) / len(sealed))
        return out

    def gold_measure(self, recipe: Recipe):
        acc = _acc(recipe)
        return {t: _correct(acc, f"gold::{t}") for t in self.tasks}

    def framing(self):
        import numpy as np
        rng = np.random.default_rng(0)
        n = 240
        X = rng.normal(size=(n, 6))
        if self.leaky_framing:
            # a gameable framing: the sealed set is so class-imbalanced that the TRIVIAL majority-class
            # baseline already clears theta (and shuffled-label fits do too) -> the meta-certifier must
            # REFUSE to let the frozen certifier drive search on it.
            y = (rng.random(n) < 0.9).astype(int)
        else:
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
        idx = np.arange(n)
        train_idx, sealed_idx = idx[:160], idx[160:]
        if self.contaminated_data:
            # copy 8 train rows verbatim into the sealed set -> near-duplicate straddle (the one leak the
            # data certificate refuses outright). The autoresearcher must never run on this dataset.
            X[sealed_idx[:8]] = X[train_idx[:8]]

        def fit_predict_fn(Xa, ya, tr, se):
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500).fit(Xa[tr], ya[tr])
            return clf.predict(Xa[se])

        def metric_fn(yt, yp):
            import numpy as _np
            return float((_np.asarray(yt) == _np.asarray(yp)).mean())

        return {"X": X, "y": y, "train_idx": train_idx, "sealed_idx": sealed_idx,
                "fit_predict_fn": fit_predict_fn, "metric_fn": metric_fn}


def _make(arena, *, driver="climb", code_authorer=None, code_prob=0.0, data_pool_root=None):
    seed = arena.seed_recipes()
    cfg = GeneratorConfig(task_hint="fine_grained", fanout=6, pool_limit=20, code_prob=code_prob)
    gen = RecipeGenerator(seed, config=cfg, seed=0, code_authorer=code_authorer, task_shape="binary")
    return RecipeResearcher(arena, gen, alpha=0.1, theta_floor=0.65, peek_budget=16,
                            competence_ceiling=0.80, driver=driver,
                            data_pool_root=data_pool_root, dataset_name="synthetic_arena")


def test_open_discovery_reaches_non_seed_backbone():
    cert = _make(SyntheticRecipeArena()).run()
    assert not cert.refused
    assert cert.novelty["menu_free"] is True, cert.novelty
    assert cert.novelty["backbone_is_novel"] is True
    assert "dinov2" in cert.champion_recipe["backbone"].lower(), cert.champion
    assert len(cert.promotions) >= 1
    # the system reached BEYOND its single seed backbone
    assert cert.discovery["n_proposed_backbones"] > cert.discovery["n_seed_backbones"]
    # gold confirmation on a never-peeked set agrees
    assert cert.gold_confirmation is not None and cert.gold_confirmation["confirmed"] is True


def test_frozen_certifier_is_sole_promoter_on_sealed_labels():
    # destroy the winner's SEALED signal only; its validation still screens through -> if promotion rode on
    # validation it would still promote dinov2. It must NOT (frozen sealed compare blocks it).
    cert = _make(SyntheticRecipeArena(flip_winner_sealed=True)).run()
    assert "dinov2" not in cert.champion_recipe["backbone"].lower(), (
        "winner with zeroed sealed labels was promoted -> promotion is NOT riding on sealed labels")


def test_meta_certifier_refuses_leaky_framing():
    cert = _make(SyntheticRecipeArena(leaky_framing=True)).run()
    assert cert.refused is True
    assert cert.framing_report is not None and cert.framing_report["trustworthy"] is False
    assert cert.peeks_used == 0


def test_authoring_sandbox_rejects_unsafe_code():
    researcher = _make(SyntheticRecipeArena())
    unsafe = Recipe(backbone="resnet18", adaptation="linear_probe",
                    code_patch="import os\ndef build(params, seed):\n    os.system('echo hi')\n",
                    code_role="classifier")
    assert researcher._admit(unsafe) is False
    # a recipe with no code patch always admits
    assert researcher._admit(Recipe(backbone="resnet18")) is True


def test_search_driver_reaches_non_seed_champion():
    cert = _make(SyntheticRecipeArena(), driver="search").run()
    assert not cert.refused
    assert cert.driver == "search"
    assert cert.novelty["menu_free"] is True, cert.novelty
    assert len(cert.promotions) >= 1


def test_datapool_gate_refuses_contaminated_dataset(tmp_path):
    # clean data is ADMITTED and recorded in the append-only pool; the run proceeds normally
    clean = _make(SyntheticRecipeArena(), data_pool_root=str(tmp_path / "clean")).run()
    assert not clean.refused
    assert clean.data_hygiene is not None and clean.data_hygiene["admitted"] is True
    assert clean.data_hygiene["certificate"]["near_dup_straddle"] == 0
    # near-duplicate train/sealed straddle (the non-negotiable leak) is REFUSED before any sealed peek
    bad = _make(SyntheticRecipeArena(contaminated_data=True), data_pool_root=str(tmp_path / "bad")).run()
    assert bad.refused is True
    assert bad.data_hygiene is not None and bad.data_hygiene["admitted"] is False
    assert bad.peeks_used == 0


def test_certificate_is_replayable_and_multiplicity_present():
    cert = _make(SyntheticRecipeArena()).run()
    # replayable champion recipe + honest session multiplicity accounting always present
    assert set(["backbone", "adaptation", "label"]).issubset(cert.champion_recipe)
    assert cert.multiplicity is not None
    assert cert.multiplicity["sealed_comparisons"] == cert.peeks_used
    assert cert.qd_coverage >= 1
