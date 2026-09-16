"""REGENERATION OPERATORS + QUALITY-DIVERSITY ARCHIVE -- turn a one-shot generator into an evolving
population.

========================================================================================================
STATUS (2026-06): WIRED into the REGENERATIVE researcher. The mutate/recombine/graft operators are the
  engine of vfplatform/recipe_generator.RecipeGenerator.expand(), and the QDArchive (MAP-Elites over
  (backbone_family, adaptation, aggregation)) is held live by RecipeResearcher to preserve recipe-space
  diversity (cert.qd_coverage). On real WILDS Camelyon17 distribution shift the regenerative loop climbed
  resnet18 -> a NON-SEED DINOv2 backbone it was never handed, gold-confirmed on a held-out hospital.

  The earlier measurement still stands and is WHY the population evolves over the RIGHT axis: diversifying
  AUTHORED recipes on a FIXED representation was measured dead (B2 authored methods vs tuned GBM 0/5 FDR;
  B1 search cycle 0/5). So the operators here mutate the OPEN recipe -- above all the backbone (the proven
  lever) -- not just a head on a frozen encoder. The frozen certifier remains the sole promoter.
========================================================================================================

WHY THIS EXISTS
---------------
Generation = one-shot, memoryless: ask the base model for "a better idea" and you regress to its prior
(textbook ML). Regeneration = a POPULATION of recipes that persists, mutates, recombines, and is re-derived
under a measured fitness. The edge over a frontier researcher is not a better idea -- it is a better loop
over ideas: parallel + diverse + selected by an un-gameable fitness.

Two pieces:
  * OPERATORS -- mutate (perturb a recipe), recombine (uniform crossover of two parents), graft (overlay a
    motif from a small 'literature' library). These are how the population moves OFF the base-model prior.
  * QD ARCHIVE -- a MAP-Elites grid keyed by a BEHAVIOR DESCRIPTOR (e.g. model family x complexity bucket).
    It keeps the best recipe PER CELL, so diversity is preserved by construction: a non-obvious region is
    never crowded out by the currently-dominant one. This is exactly the Pareto/diversity machine that
    surfaces a winner a greedy, prior-following search would never reach.

THE INVARIANT
-------------
The archive's `fitness` is supplied by the caller and, in the live system, comes from the verification
cascade (ultimately the frozen Tier-3 certificate's lower bound). The archive ORGANIZES search; it never
certifies and never promotes. Feeding it a leaked/gamed fitness is the caller's bug, not a path around the
frozen gate. Deterministic given a seeded RNG.

CONTRACT: stdlib + a typed gene space (no estimator, no certifier import). GeneValue is a precise union,
not an opaque type.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

GeneValue = Union[str, int, float, bool]
Genome = Dict[str, GeneValue]


@dataclass
class Param:
    """One dimension of the recipe space: either CATEGORICAL (choices) or NUMERIC (low..high, optional int)."""
    name: str
    choices: Optional[List[GeneValue]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    is_int: bool = False

    def __post_init__(self):
        if self.choices is None and (self.low is None or self.high is None):
            raise ValueError(f"Param {self.name!r} must be categorical (choices) or numeric (low/high)")
        if self.choices is not None and len(self.choices) == 0:
            raise ValueError(f"Param {self.name!r} has empty choices")

    @property
    def is_categorical(self) -> bool:
        return self.choices is not None

    def sample(self, rng: random.Random) -> GeneValue:
        if self.is_categorical:
            return rng.choice(self.choices)
        v = rng.uniform(float(self.low), float(self.high))
        return int(round(v)) if self.is_int else v

    def mutate(self, value: GeneValue, rng: random.Random) -> GeneValue:
        if self.is_categorical:
            others = [c for c in self.choices if c != value]
            return rng.choice(others) if others else value
        span = float(self.high) - float(self.low)
        nv = float(value) + rng.gauss(0.0, 0.2 * span)
        nv = min(max(nv, float(self.low)), float(self.high))
        return int(round(nv)) if self.is_int else nv


class Space:
    """The recipe space: an ordered set of Params. Samples and validates whole genomes."""

    def __init__(self, params: Sequence[Param]):
        if not params:
            raise ValueError("Space needs at least one Param")
        self.params: Dict[str, Param] = {p.name: p for p in params}

    def sample(self, rng: random.Random) -> Genome:
        return {name: p.sample(rng) for name, p in self.params.items()}

    def clamp(self, genome: Genome) -> Genome:
        """Coerce a genome back into the space (used after graft of a partial motif)."""
        out: Genome = {}
        for name, p in self.params.items():
            if name in genome:
                out[name] = genome[name]
            else:
                out[name] = p.sample(random.Random(0))
        return out


# -- operators ------------------------------------------------------------------------------------------

def mutate(genome: Genome, space: Space, rng: random.Random, *, rate: float = 0.3,
           single_gene: bool = False) -> Genome:
    """Perturb a recipe. With single_gene=True, change EXACTLY one gene (standard hill-climb step); else
    each gene mutates independently with probability `rate` (the bolder regeneration step)."""
    child = dict(genome)
    names = list(space.params)
    if single_gene:
        name = rng.choice(names)
        child[name] = space.params[name].mutate(genome.get(name, space.params[name].sample(rng)), rng)
        return child
    for name in names:
        if rng.random() < rate:
            child[name] = space.params[name].mutate(genome.get(name, space.params[name].sample(rng)), rng)
    return child


def recombine(g1: Genome, g2: Genome, rng: random.Random) -> Genome:
    """Uniform crossover: each gene independently inherited from one of the two parents."""
    child: Genome = {}
    for name in set(g1) | set(g2):
        src = g1 if (name in g1 and (name not in g2 or rng.random() < 0.5)) else g2
        child[name] = src[name]
    return child


@dataclass
class Motif:
    """A named 'literature' template: a PARTIAL genome (a known-good motif) to graft onto a recipe."""
    name: str
    genes: Genome


class LiteratureLibrary:
    """A small library of motifs the proposer can graft -- the 'recombine with known good ideas' operator."""

    def __init__(self, motifs: Optional[Sequence[Motif]] = None):
        self.motifs: List[Motif] = list(motifs or [])

    def add(self, motif: Motif) -> None:
        self.motifs.append(motif)

    def graft(self, genome: Genome, rng: random.Random, space: Optional[Space] = None) -> Genome:
        """Overlay a randomly chosen motif's genes onto `genome` (motif wins on conflict). No-op if empty."""
        if not self.motifs:
            return dict(genome)
        motif = rng.choice(self.motifs)
        child = dict(genome)
        child.update(motif.genes)
        return space.clamp(child) if space is not None else child


# -- quality-diversity archive --------------------------------------------------------------------------

Descriptor = Tuple[GeneValue, ...]


@dataclass
class Elite:
    genome: Genome
    fitness: float
    descriptor: Descriptor


class QDArchive:
    """MAP-Elites archive: keeps the highest-fitness recipe per behavior-descriptor cell. Preserves
    diversity so a non-obvious region is not crowded out by the dominant one."""

    def __init__(self):
        self.cells: Dict[Descriptor, Elite] = {}

    def add(self, genome: Genome, fitness: float, descriptor: Descriptor) -> bool:
        """Insert if the cell is empty or this recipe beats the cell's incumbent. Returns True on insert."""
        cur = self.cells.get(descriptor)
        if cur is None or fitness > cur.fitness:
            self.cells[descriptor] = Elite(dict(genome), float(fitness), descriptor)
            return True
        return False

    def coverage(self) -> int:
        return len(self.cells)

    def elites(self) -> List[Elite]:
        return sorted(self.cells.values(), key=lambda e: -e.fitness)

    def best(self) -> Optional[Elite]:
        elites = self.elites()
        return elites[0] if elites else None

    def select_parents(self, rng: random.Random, k: int = 2) -> List[Genome]:
        """Sample parents UNIFORMLY over occupied cells (not over fitness) -- this is what preserves
        diversity: a lonely high-potential region is sampled as often as the crowded dominant one."""
        if not self.cells:
            return []
        cells = list(self.cells.values())
        return [cells[rng.randrange(len(cells))].genome for _ in range(k)]


def regenerate(space: Space, fitness_fn: Callable[[Genome], float],
               descriptor_fn: Callable[[Genome], Descriptor], *, generations: int = 30,
               batch: int = 12, init_size: int = 12, rng: Optional[random.Random] = None,
               library: Optional[LiteratureLibrary] = None,
               archive: Optional[QDArchive] = None) -> QDArchive:
    """Run the regeneration loop: initialize the archive with random recipes, then for each generation
    select diverse parents, apply operators (mutate / recombine / graft), evaluate fitness, and update the
    archive. Returns the QD archive (best() is the surfaced winner). fitness_fn is the measured fitness
    (in the live system, the cascade's certified lower bound)."""
    rng = rng or random.Random(0)
    arc = archive if archive is not None else QDArchive()
    for _ in range(init_size):
        g = space.sample(rng)
        arc.add(g, fitness_fn(g), descriptor_fn(g))
    for _ in range(generations):
        for _ in range(batch):
            roll = rng.random()
            if roll < 0.5 or arc.coverage() < 2:
                parents = arc.select_parents(rng, k=1)
                child = mutate(parents[0], space, rng) if parents else space.sample(rng)
            elif roll < 0.8:
                p = arc.select_parents(rng, k=2)
                child = recombine(p[0], p[1], rng)
            else:
                p = arc.select_parents(rng, k=1)
                base = p[0] if p else space.sample(rng)
                child = library.graft(base, rng, space) if library is not None else mutate(base, space, rng)
            arc.add(child, fitness_fn(child), descriptor_fn(child))
    return arc


def greedy_hillclimb(space: Space, fitness_fn: Callable[[Genome], float], *, start: Genome,
                     steps: int = 200, rng: Optional[random.Random] = None) -> Tuple[Genome, float]:
    """Single-gene hill-climb from `start`, accepting only improvements. This is the base-prior baseline:
    it cannot cross a fitness valley, so it gets stuck in the locally-dominant region."""
    rng = rng or random.Random(0)
    cur, cur_f = dict(start), fitness_fn(start)
    for _ in range(steps):
        cand = mutate(cur, space, rng, single_gene=True)
        f = fitness_fn(cand)
        if f > cur_f:
            cur, cur_f = cand, f
    return cur, cur_f


__all__ = ["GeneValue", "Genome", "Param", "Space", "Motif", "LiteratureLibrary", "Elite", "QDArchive",
           "Descriptor", "mutate", "recombine", "regenerate", "greedy_hillclimb"]
