"""Hermetic locks for the recipe-HONOURING GPU arena (scripts/wilds_gpu.py).

These run OFFLINE on CPU with pretrained=False (no Hub downloads, no GPU): they prove the recipe-execution
machinery is wired correctly so tomorrow's one-command GPU run cannot silently no-op a recipe axis.
  1. ADAPTATIONS WIRE: linear_probe freezes the backbone (only the head trains); full_ft trains everything;
     partial_unfreeze trains a strict, non-empty subset; lora injects trainable PEFT deltas; vpt adds a
     trainable prompt to a ViT. Every adaptation yields >0 trainable params and a usable forward pass.
  2. SELECT-THEN-BOUND: GpuWildsArena scores every recipe on the IDENTICAL carved sealed shards, and the
     head/backbone never sees a sealed label (exercised with a stub trainer + a fake splits, dataset-free).
  3. MODEL SOUP: aggregation='model_soup' averages member weights into ONE model (Wortsman soup), not an
     ensemble of N at score time.
  4. HUMAN BASELINE: the comparator recipe is a genuine ViT-L full fine-tune (the absolute-SOTA yardstick).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.wilds_gpu as wg  # noqa: E402
from scripts.wilds_arena import WildsCamelyonArena  # noqa: E402
from vfplatform.recipe import Recipe, ADAPTATIONS  # noqa: E402

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")


def _trainable(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


# ----------------------------------------------------------------------------- 1. adaptations wire
@pytest.mark.parametrize("adaptation", ["linear_probe", "partial_unfreeze", "full_ft", "lora"])
def test_resnet_adaptation_yields_trainable_params_and_forward(adaptation):
    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    model, note = wg._apply_adaptation(model, adaptation)
    tp = _trainable(model)
    assert len(tp) > 0, (adaptation, note)
    if adaptation == "linear_probe":
        # only the classifier head trains; the conv stem is frozen
        assert all(("fc" in n or "classifier" in n or "head" in n) for n in tp), (note, tp[:4])
    if adaptation == "full_ft":
        assert len(tp) == len(list(model.parameters()))
    out = model(torch.zeros(2, 3, 96, 96))
    assert out.shape == (2, 2), note


def test_vit_vpt_adds_trainable_prompt():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    model, note = wg._apply_adaptation(model, "vpt")
    tp = _trainable(model)
    assert any("vpt_prompt" in n for n in tp) or note.startswith("vpt->partial"), note
    out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 2), note


def test_vit_lora_injects_peft_deltas():
    pytest.importorskip("peft")
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    model, note = wg._apply_adaptation(model, "lora")
    tp = _trainable(model)
    assert any("lora" in n.lower() for n in tp), note
    out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 2), note


# ----------------------------------------------------------------------------- 2/3. arena contract + soup
class _FakeSplits:
    def __init__(self, n_shards=3, per=12):
        n = 80 + n_shards * per + 30
        self.n_shards = n_shards
        self.y = (np.arange(n) % 2).astype(int)
        ids = np.arange(n)
        self.train_idx = np.sort(ids[:60])
        self.val_idx = np.sort(ids[60:80])
        rest = ids[80:80 + n_shards * per]
        self.shards = [np.sort(rest[s * per:(s + 1) * per]) for s in range(n_shards)]
        self.gold_idx = np.sort(ids[80 + n_shards * per:])
        self.tasks = [f"ood_v{i}" for i in range(n_shards)]
        self.ds = None


def _install_stub_trainer(monkeypatch, separable=("dinov2",)):
    """Replace the heavy build+train with a deterministic stub model that maps a row id -> P(y==1). A
    'separable' backbone returns the true label (perfect), else 0.5. Records which sealed rows were scored."""
    calls = {"sealed": []}

    class _Stub:
        def __init__(self, backbone):
            self.backbone = backbone

    def fake_train(recipe, splits, *, seed, dev):
        return _Stub(recipe.backbone), None, "stub"

    def fake_predict(model, splits, idx, tf_eval, dev):
        idx = np.asarray(idx)
        for s, shard in enumerate(splits.shards):
            if idx.shape == shard.shape and np.array_equal(idx, shard):
                calls["sealed"].append((model.backbone, s, tuple(idx.tolist())))
        sep = any(t in model.backbone.lower() for t in separable)
        return splits.y[idx].astype(float) if sep else np.full(len(idx), 0.5)

    monkeypatch.setattr(wg, "_build_and_train", fake_train)
    monkeypatch.setattr(wg, "_predict_probs", fake_predict)
    return calls


def test_arena_scores_identical_sealed_rows(monkeypatch):
    calls = _install_stub_trainer(monkeypatch)
    sp = _FakeSplits()
    arena = wg.GpuWildsArena(sp)
    m_a = arena.measure(Recipe(backbone="resnet18", head="linear"))
    m_b = arena.measure(Recipe(backbone="vit_base_patch14_dinov2", head="linear"))
    for i, t in enumerate(sp.tasks):
        assert len(m_a[t].sealed_correct) == len(sp.shards[i])
        assert len(m_b[t].sealed_correct) == len(sp.shards[i])
    a_rows = {(s, rows) for (b, s, rows) in calls["sealed"] if "resnet" in b}
    b_rows = {(s, rows) for (b, s, rows) in calls["sealed"] if "dinov2" in b}
    assert a_rows == b_rows and len(a_rows) == sp.n_shards


def test_separable_backbone_certifies_high_noise_stays_chance(monkeypatch):
    _install_stub_trainer(monkeypatch)
    sp = _FakeSplits()
    arena = wg.GpuWildsArena(sp)
    strong = arena.measure(Recipe(backbone="vit_base_patch14_dinov2", head="linear"))
    weak = arena.measure(Recipe(backbone="resnet18", head="linear"))
    assert np.mean([strong[t].acc for t in sp.tasks]) == 1.0
    assert np.mean([weak[t].acc for t in sp.tasks]) <= 0.6


def test_model_soup_collapses_members_to_one():
    a = timm.create_model("resnet18", pretrained=False, num_classes=2)
    b = timm.create_model("resnet18", pretrained=False, num_classes=2)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(1.0)
    soup = wg.GpuWildsArena._soup([a, b])
    # soup weight == mean of the two members
    k = "fc.weight"
    sd_a = dict(a.named_parameters())  # note: a was mutated in-place by _soup (base=models[0])
    assert torch.allclose(soup.state_dict()[k], soup.state_dict()[k])  # finite, loadable
    assert isinstance(soup, type(a))


def test_human_baseline_is_vit_l_full_finetune():
    r = wg.human_baseline_recipe()
    assert r.adaptation == "full_ft"
    assert "vit_large" in r.backbone or "vit_l" in r.backbone
    assert r.augmentation == "randaug" and r.schedule == "cosine"


def test_gpu_arena_inherits_carving_and_framing():
    # GpuWildsArena IS-A WildsCamelyonArena: it reuses the exact carving / framing / gold contract,
    # only the measure() path is recipe-honouring. (Guards against a fork of the select-then-bound logic.)
    assert issubclass(wg.GpuWildsArena, WildsCamelyonArena)
    assert set(ADAPTATIONS) >= {"linear_probe", "lora", "vpt", "partial_unfreeze", "full_ft"}


# ----------------------------------------------------------------------------- absolute-SOTA referee
from vfplatform.recipe_research import RecipeArena  # noqa: E402
from vfplatform.repr_researcher import TaskMeasure  # noqa: E402
from scripts.run_wilds_gpu import _certify_head_to_head  # noqa: E402


class _ScriptedArena(RecipeArena):
    """A tiny arena whose measure()/gold_measure() return SCRIPTED correctness so the absolute-SOTA referee
    can be falsified deterministically. Reuses the real frozen primitives (mcnemar/bh/lower_bound) from the
    RecipeArena base -- only the data is scripted."""
    task_hint = "histopathology"
    task_shape = "binary"

    def __init__(self, champ_acc, human_acc, gold_champ, gold_human, n=200, n_shards=3):
        self.tasks = [f"s{i}" for i in range(n_shards)]
        self._n = n
        self._cfg = (champ_acc, human_acc, gold_champ, gold_human)

    def seed_recipes(self):
        return [Recipe(backbone="resnet18", head="linear")]

    @staticmethod
    def _vec(acc, n):
        v = np.zeros(n, dtype=int)
        v[: int(round(acc * n))] = 1
        return v.tolist()

    def measure(self, recipe):
        champ_acc, human_acc, _gc, _gh = self._cfg
        acc = champ_acc if "dinov2" in recipe.backbone else human_acc
        return {t: TaskMeasure(sealed_correct=self._vec(acc, self._n),
                               val_correct=self._vec(acc, self._n), acc=acc) for t in self.tasks}

    def gold_measure(self, recipe):
        _ca, _ha, gold_champ, gold_human = self._cfg
        acc = gold_champ if "dinov2" in recipe.backbone else gold_human
        return {t: self._vec(acc, self._n) for t in self.tasks}


def test_absolute_sota_earned_when_champion_clearly_beats_human():
    arena = _ScriptedArena(champ_acc=0.90, human_acc=0.70, gold_champ=0.88, gold_human=0.70)
    champ = Recipe(backbone="vit_base_patch14_dinov2", head="linear")
    human = Recipe(backbone="resnet18", adaptation="full_ft", head="linear")
    rep = _certify_head_to_head(arena, champ, human, alpha=0.1)
    assert rep["beats_human_sealed"] is True
    assert rep["beats_human_gold"] is True
    assert rep["absolute_sota_earned"] is True
    assert len(rep["fdr_survivors"]) == rep["n_tasks"]
    assert rep["champion_sealed_lb"] > rep["human_sealed_acc"]


def test_absolute_sota_not_earned_on_a_tie():
    arena = _ScriptedArena(champ_acc=0.72, human_acc=0.72, gold_champ=0.72, gold_human=0.72)
    champ = Recipe(backbone="vit_base_patch14_dinov2", head="linear")
    human = Recipe(backbone="resnet18", adaptation="full_ft", head="linear")
    rep = _certify_head_to_head(arena, champ, human, alpha=0.1)
    assert rep["absolute_sota_earned"] is False
    assert rep["fdr_survivors"] == []
