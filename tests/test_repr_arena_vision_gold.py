"""HERMETIC test for the vision (FGVC-Aircraft) arena's GOLD contract.

The gold-confirmation tier requires a partition DISJOINT from the train/val/sealed rows. FGVC-Aircraft pools
trainval+test to exactly 100 images/variant, and the arena's per_class=100 consumes every one of them, so no
disjoint gold set exists. The honest, leak-safe behaviour is therefore that the arena reports NO gold set
(gold_measure -> None) rather than fabricating an overlapping one. That contract is pure (no disk, no network,
no weights) and is locked here; the live gold tier is exercised on the text arena, which has corpus headroom.
"""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}

_PATH = os.path.join(_ROOT, "scripts", "repr_arena.py")
_spec = importlib.util.spec_from_file_location("repr_arena", _PATH)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def test_vision_arena_honestly_reports_no_gold_set():
    """At per_class=100 the FGVC arena has no disjoint headroom, so gold_measure returns None for any
    principal (champion, baseline, or fusion) -- the brain then emits no fabricated gold confirmation."""
    arena = V.FgvcAircraftArena(smoke=1)        # __init__ touches no disk/network
    assert arena.gold_measure(V.BASELINE_TAG) is None
    assert arena.gold_measure("dinov2_g") is None
    assert arena.gold_measure("fuse[dinov2_vitl14+clip_vitb32]") is None


def test_arena_advertises_the_gold_interface():
    """gold_measure is part of the Arena interface the brain calls; the vision arena must implement it (even if
    it returns None) so the routing layer can treat every arena uniformly."""
    assert hasattr(V.FgvcAircraftArena, "gold_measure")
