"""Tests for regeneration operators + QD archive.

Acceptance (Phase 6): on a DECEPTIVE landscape where the base prior favors family 'A' (a local optimum)
but the true winner lives in family 'C' (reachable only by crossing a fitness valley), the QD archive
surfaces the non-obvious 'C' winner while a base-prior greedy hill-climb stays stuck on 'A'."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import regeneration as RG


# Recipe space: family in {A,B,C}, depth 1..10, feature flag in {plain, special}.
SPACE = RG.Space([
    RG.Param("family", choices=["A", "B", "C"]),
    RG.Param("depth", low=1, high=10, is_int=True),
    RG.Param("feature", choices=["plain", "special"]),
])


def fitness(g: RG.Genome) -> float:
    """Deceptive: A is a broad, safe local optimum (~0.72). C is mediocre by default (~0.58) UNLESS it
    also has the 'special' feature and depth in [4,6], where it jumps to ~0.95. So a single-gene climber
    starting at A cannot reach C's peak (flipping to C alone DROPS fitness -> rejected)."""
    fam, depth, feat = g["family"], int(g["depth"]), g["feature"]
    if fam == "A":
        return 0.72 - 0.01 * abs(depth - 5)
    if fam == "B":
        return 0.50
    # family C
    if feat == "special" and 4 <= depth <= 6:
        return 0.95 - 0.01 * abs(depth - 5)
    return 0.58


def descriptor(g: RG.Genome) -> RG.Descriptor:
    return (g["family"],)        # one cell per family -> C is preserved no matter how dominant A is


def test_qd_surfaces_nonobvious_winner_vs_greedy():
    rng = random.Random(0)
    arc = RG.regenerate(SPACE, fitness, descriptor, generations=40, batch=16, init_size=16, rng=rng)
    best = arc.best()
    assert best is not None
    # QD found the C-region peak
    assert best.genome["family"] == "C" and best.fitness > 0.9
    # all three families are represented (diversity preserved)
    assert arc.coverage() == 3

    # base-prior greedy hill-climb starting at A's optimum stays stuck well below the C peak
    start = {"family": "A", "depth": 5, "feature": "plain"}
    _, greedy_f = RG.greedy_hillclimb(SPACE, fitness, start=start, steps=300, rng=random.Random(1))
    assert greedy_f <= 0.73
    assert best.fitness > greedy_f + 0.15      # QD decisively beats the base-prior climber


def test_operators_produce_valid_genomes():
    rng = random.Random(2)
    g1, g2 = SPACE.sample(rng), SPACE.sample(rng)
    for child in (RG.mutate(g1, SPACE, rng), RG.recombine(g1, g2, rng)):
        assert child["family"] in ["A", "B", "C"]
        assert 1 <= int(child["depth"]) <= 10
        assert child["feature"] in ["plain", "special"]


def test_single_gene_mutation_changes_exactly_one():
    rng = random.Random(3)
    g = {"family": "A", "depth": 5, "feature": "plain"}
    child = RG.mutate(g, SPACE, rng, single_gene=True)
    diffs = sum(1 for k in g if g[k] != child[k])
    assert diffs == 1


def test_literature_graft_overlays_motif():
    lib = RG.LiteratureLibrary([RG.Motif("c_special", {"family": "C", "feature": "special"})])
    base = {"family": "A", "depth": 5, "feature": "plain"}
    child = lib.graft(base, random.Random(0), SPACE)
    assert child["family"] == "C" and child["feature"] == "special" and int(child["depth"]) == 5


def test_qd_archive_keeps_best_per_cell():
    arc = RG.QDArchive()
    assert arc.add({"family": "C", "depth": 5, "feature": "plain"}, 0.58, ("C",)) is True
    assert arc.add({"family": "C", "depth": 5, "feature": "special"}, 0.95, ("C",)) is True   # improves
    assert arc.add({"family": "C", "depth": 3, "feature": "plain"}, 0.40, ("C",)) is False     # worse
    assert arc.cells[("C",)].fitness == 0.95


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
