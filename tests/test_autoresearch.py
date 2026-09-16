"""Hermetic locks for the unified autoresearch entrypoint (scripts/run_autoresearch.py).

The brain's autonomous-promotion invariant and each arena's leak-freeness are already locked elsewhere
(tests/test_repr_researcher.py, tests/test_text_arena.py). What this file locks is the ROUTING/AGGREGATION
layer that the one-command entrypoint adds on top -- with lightweight fake certificates, no caches, no network:

  * the per-modality acceptance predicates are honest (they FAIL when the verdict is not reproduced / not
    gold-confirmed, PASS when it is);
  * the cross-modal session certificate is a faithful AND over the modalities (the law only holds if the lever
    fires in every modality, the champion is multiplicity-robust everywhere, and gold confirms where it exists),
    and it correctly SKIPS gold for arenas that honestly supply none (gold_confirmed=None).
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_autoresearch import (  # noqa: E402
    FROZEN_EXPECTED, _accept_text, _accept_vision, _cert_dict, _session_certificate,
)

GOOD = FROZEN_EXPECTED
BAD = {"vectorforge/science.py": "deadbeef", "vfplatform/sealed.py": "30ad6245"}


def _promo(rung, frm, to, survivors, lift):
    return SimpleNamespace(rung=rung, from_tag=frm, to_tag=to, survivors=survivors, n=10, mean_lift=lift)


def _rej(rung, tag):
    return SimpleNamespace(rung=rung, tag=tag, survivors=[], n=10, mean_lift=0.0, reason="")


def _vision_cert(champion="dinov2_g"):
    return SimpleNamespace(
        champion=champion, champion_family="dinov2",
        promotions=[_promo("model", "clip_vitb32", "dinov2_vitl14", ["a", "b"], 0.10),
                    _promo("capacity", "dinov2_vitl14", "dinov2_g", ["c"], 0.05)],
        rejections=[_rej("model", "siglip_so"), _rej("model", "eva02_l")],
        data_ceiling_tasks=["737-700_vs_737-800", "737-300_vs_737-400"])


def _text_cert(gold_confirmed=True, family="mpnet"):
    gc = {"confirmed": gold_confirmed, "champion": "mpnet", "baseline": "tfidf_lsa",
          "gold_n": 2400, "n_tasks": 6, "survivors": ["x"] * 5, "mean_lift": 0.063}
    return SimpleNamespace(
        champion="mpnet", champion_family=family,
        promotions=[_promo("model", "tfidf_lsa", "mpnet", ["a", "b", "c", "d"], 0.081)],
        rejections=[], data_ceiling_tasks=[], gold_confirmation=gc)


def test_vision_acceptance_requires_the_pre_registered_verdict():
    assert all(_accept_vision(_vision_cert(), GOOD).values())                 # reproduces the verdict -> PASS
    assert not _accept_vision(_vision_cert(champion="siglip_so"), GOOD)["champion is DINOv2-g"]
    assert not _accept_vision(_vision_cert(), BAD)["frozen certifier byte-identical"]


def test_text_acceptance_requires_a_gold_confirmed_neural_champion():
    assert all(_accept_text(_text_cert(), GOOD, "tfidf_lsa").values())        # gold-confirmed neural -> PASS
    no_gold = _accept_text(_text_cert(gold_confirmed=False), GOOD, "tfidf_lsa")
    assert not no_gold["champion CONFIRMED on a NEVER-PEEKED gold set"]
    lexical = _accept_text(_text_cert(family="lexical"), GOOD, "tfidf_lsa")
    assert not lexical["champion is a frozen neural encoder (not lexical)"]


def _result(modality, cert, multiplicity_robust, all_pass=True):
    cfg = {"label": f"{modality}-arena", "out": f"{modality}.json"}
    checks = {"ok": all_pass}
    mp = {"robust_to_session_multiplicity": multiplicity_robust, "sealed_comparisons": 5,
          "mcnemar_tests_total": 50, "per_comparison_fdr_alpha": 0.1, "session_bonferroni_alpha": 0.02,
          "fdr_survivors_nominal": ["a"], "bonferroni_survivors_session": ["a"] if multiplicity_robust else [],
          "n_tasks": 10, "gold_independent_confirmation": False}
    cert = SimpleNamespace(
        champion=cert.champion, champion_family=cert.champion_family,
        move_class_path=["model"], stop_reason="stop", peeks_used=5,
        promotions=cert.promotions, rejections=cert.rejections,
        data_ceiling_tasks=cert.data_ceiling_tasks,
        sealed_acc={}, sealed_lb={}, pareto_front=[], pareto_report="",
        gold_confirmation=getattr(cert, "gold_confirmation", None), multiplicity=mp, log=[])
    return _cert_dict(cert, modality, cfg, checks, GOOD)


def test_cert_dict_carries_multiplicity_gold_and_acceptance():
    d = _result("text", _text_cert(), True)
    assert d["multiplicity"]["robust_to_session_multiplicity"] is True
    assert d["gold_confirmation"]["confirmed"] is True
    assert d["all_pass"] is True and d["frozen_hashes"] == GOOD


def test_session_certificate_is_an_honest_and_over_modalities():
    vis = _result("vision", _vision_cert(), multiplicity_robust=True)          # vision honestly has no gold
    vis["gold_confirmation"] = None
    txt = _result("text", _text_cert(), multiplicity_robust=True)
    sess = _session_certificate([vis, txt])
    assert sess["representation_lever_fires_in_every_modality"] is True
    assert sess["champion_robust_to_session_multiplicity_everywhere"] is True
    assert sess["champion_gold_confirmed_where_gold_exists"] is True          # skips vision's None
    assert sess["per_modality"]["vision"]["gold_confirmed"] is None
    assert sess["all_modalities_accept"] is True


def test_session_certificate_fails_when_one_modality_is_not_robust():
    vis = _result("vision", _vision_cert(), multiplicity_robust=False)
    txt = _result("text", _text_cert(), multiplicity_robust=True)
    sess = _session_certificate([vis, txt])
    assert sess["champion_robust_to_session_multiplicity_everywhere"] is False
    assert sess["representation_lever_fires_in_every_modality"] is True


def test_session_certificate_fails_when_gold_is_present_but_unconfirmed():
    txt_bad = _result("text", _text_cert(gold_confirmed=False), multiplicity_robust=True)
    sess = _session_certificate([txt_bad])
    assert sess["champion_gold_confirmed_where_gold_exists"] is False
