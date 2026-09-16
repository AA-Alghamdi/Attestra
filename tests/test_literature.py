"""Hermetic locks for the LITERATURE-GROUNDED discovery layer (vfplatform/literature.py + its wiring into
the RecipeGenerator / DiscoveryLedger).

These run with NO network: every connector is forced offline (monkeypatched HTTP returns None) or driven
from the bundled corpus / injected fake findings. What they lock:

  * connectors are OFFLINE-GRACEFUL -- any HTTP failure yields [] (never raises), and retrieve() falls back
    to the bundled corpus so downstream is never starved;
  * extraction is faithful -- verbatim Hub ids survive, family mentions canonicalize to one concrete id,
    and technique keywords distil into typed motifs;
  * the typed firewall (validate_genes) DROPS off-axis keys/values, so a hallucinated/LLM motif can never
    inject an illegal recipe;
  * the LLM bridge returns None without a key (deterministic extractor takes over);
  * the ANTI-MENU wiring is real and FALSIFIABLE -- a literature backbone enters the pool and is tagged in
    the ledger, a champion that uses it is `literature_grounded=True` with a traceable source, and a
    champion that uses a SEED backbone is `literature_grounded=False`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import literature as L  # noqa: E402
from vfplatform.recipe import Recipe  # noqa: E402
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator  # noqa: E402
from vfplatform.regeneration import Motif  # noqa: E402


# ------------------------------------------------------------------ connectors are offline-graceful
def test_connectors_never_raise_when_http_is_down(monkeypatch):
    monkeypatch.setattr(L, "_http_get", lambda *a, **k: None)
    assert L.fetch_arxiv("anything") == []
    assert L.fetch_github("anything") == []
    assert L.fetch_paperswithcode("anything") == []
    # HF tries the hub client then the JSON API; force BOTH to fail to simulate true offline
    monkeypatch.setattr(L, "_get_json", lambda *a, **k: None)
    monkeypatch.setitem(os.environ, "HF_TOKEN", "")
    try:
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "list_models",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    except Exception:
        pass
    assert L.fetch_huggingface("anything") == []


def test_retrieve_falls_back_to_bundled_corpus(monkeypatch):
    monkeypatch.setattr(L, "_http_get", lambda *a, **k: None)
    monkeypatch.setattr(L, "_get_json", lambda *a, **k: None)
    try:  # the HF hub client bypasses _http_get -> force it offline too
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "list_models",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    except Exception:
        pass
    findings = L.retrieve("tumor classification under shift", online=True, use_cache=False)
    assert findings, "retrieve must never return empty -- corpus is the floor"
    assert all(isinstance(f, L.Finding) for f in findings)
    assert {f.source for f in findings} == {"corpus"}


# ------------------------------------------------------------------ extraction
def test_extract_backbones_keeps_verbatim_hub_ids_and_canonicalizes_families():
    findings = [
        L.Finding(source="huggingface", ident="facebook/dinov2-base", title="x",
                  url="u", text="image-feature-extraction", model_ids=["facebook/dinov2-base"]),
        L.Finding(source="arxiv", ident="2203.05482", title="Model soups",
                  url="u", text="we use a ConvNeXt backbone and average the weights"),
    ]
    pairs = L.extract_backbones(findings)
    ids = [mid for mid, _ in pairs]
    assert "facebook/dinov2-base" in ids                  # verbatim Hub id survives
    assert any("convnext" in mid for mid in ids)          # family mention canonicalized to a concrete id
    # provenance is preserved: every id carries the finding that surfaced it
    for mid, f in pairs:
        assert isinstance(f, L.Finding)


def test_extract_motifs_are_typed_and_provenanced():
    findings = [
        L.Finding(source="arxiv", ident="2106.09685", title="LoRA",
                  url="u", text="low-rank adaptation freezes the backbone and learns low-rank updates"),
        L.Finding(source="arxiv", ident="2203.05482", title="Model soups",
                  url="u", text="averaging the weights of multiple fine-tuned models"),
    ]
    motifs = L.extract_motifs(findings)
    names = {m.name for m, _ in motifs}
    assert "lit:lora" in names and "lit:model_soup" in names
    for m, srcs in motifs:
        assert isinstance(m, Motif) and m.genes          # non-empty validated genome
        assert srcs and all(isinstance(s, L.Finding) for s in srcs)


def test_validate_genes_drops_off_axis_keys_and_values():
    dirty = {
        "adaptation": "lora",            # legal
        "augmentation": "telepathy",     # illegal value -> dropped
        "optimizer": "adamw",            # legal
        "backbone": "vit_huge",          # not a motif axis -> dropped
        "epochs": 999,                   # clamped to 40
        "llrd": "yes",                   # coerced to bool
        "made_up_axis": 1,               # unknown -> dropped
    }
    clean = L.validate_genes(dirty)
    assert clean["adaptation"] == "lora"
    assert clean["optimizer"] == "adamw"
    assert "augmentation" not in clean
    assert "backbone" not in clean
    assert "made_up_axis" not in clean
    assert clean["epochs"] == 40
    assert clean["llrd"] is True


# ------------------------------------------------------------------ LLM bridge
def test_llm_bridge_returns_none_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert L.llm_motifs([L.Finding("arxiv", "1", "t", "u", "lora")], "p", api_key=None) is None


# ------------------------------------------------------------------ caching round-trip
def test_cache_round_trips(tmp_path, monkeypatch):
    fake = [L.Finding(source="arxiv", ident="42", title="t", url="u", text="lora")]
    monkeypatch.setattr(L, "fetch_arxiv", lambda q, limit=8: fake)
    L._CONNECTORS["arxiv"] = L.fetch_arxiv  # rebind to the patched fn
    out1 = L.retrieve("q", sources=("arxiv",), online=True, cache_dir=str(tmp_path))
    assert [f.ident for f in out1] == ["42"]
    # second read with online=False must hit the cache (not the corpus fallback)
    out2 = L.retrieve("q", sources=("arxiv",), online=False, cache_dir=str(tmp_path))
    assert [f.ident for f in out2] == ["42"]


# ------------------------------------------------------------------ anti-menu wiring (the headline)
def _scout_offline():
    # online=False + no cache => the scout builds from the bundled corpus, fully deterministic.
    return L.LiteratureScout("histopathology tumor classification distribution shift", online=False,
                             use_llm=False)


def test_literature_backbone_enters_pool_and_is_tagged():
    sc = _scout_offline()
    assert sc.backbones(), "offline corpus must surface at least one backbone"
    gen = RecipeGenerator([Recipe(backbone="resnet18")],
                          config=GeneratorConfig(task_hint="vision", pool_limit=8,
                                                 online_discovery=False),
                          scout=sc, seed=0)
    # every literature id is in the open pool AND tagged in the ledger as literature-sourced
    for mid in sc.backbones():
        assert mid in gen.pool
        assert mid in gen.ledger.literature_backbones
    # the grafted library is the RETRIEVED one, not the hard-coded default
    assert all(m.name.startswith("lit:") for m in gen.library.motifs)


def test_champion_grounding_is_falsifiable():
    sc = _scout_offline()
    gen = RecipeGenerator([Recipe(backbone="resnet18")],
                          config=GeneratorConfig(task_hint="vision", pool_limit=8,
                                                 online_discovery=False),
                          scout=sc, seed=0)
    # a champion that uses a LITERATURE backbone -> grounded, with a real traceable source
    lit_champ = Recipe(backbone=sc.backbones()[0])
    nov = gen.ledger.novelty(lit_champ)
    assert nov["literature_grounded"] is True
    src = nov["literature_source"]
    assert src and src.get("url") and src.get("source")
    # a champion that uses the SEED backbone -> NOT grounded (the guard can say 'still a menu')
    seed_nov = gen.ledger.novelty(Recipe(backbone="resnet18"))
    assert seed_nov["literature_grounded"] is False
    assert seed_nov["literature_source"] is None
