"""Harness fabric core (Phase 3): the data->Task adapter layer that is self-certifying.

# === WIRING ===
# The integrator plugs this fabric in FRONT of the Phase-0 engine. A harness turns raw,
# modality-specific data into the frozen `frontier.task.Task` that ResearchEngine.run already
# accepts, and it does so only AFTER it has proven (on a known-good built-in dataset) that the
# adapter + split + baseline path it produces actually certifies through the SAME frozen gate.
#
# Call site (what the integrator writes):
#
#     from frontier.harness import lookup, HarnessRegistry          # registry + lookup
#     from frontier.engine import ResearchEngine, EngineConfig
#
#     harness = lookup("tabular")                                   # by task-type key
#     ok, self_cert = harness.self_test()                           # GATE: trust nothing until True
#     assert ok, f"harness failed self-test: {self_cert}"
#
#     task = harness.adapt(X, y, kind="classification", theta=0.85, name="my_data")
#     #  -> task is a frontier.task.Task; theta/metric chosen by the harness for the kind
#     result = ResearchEngine(EngineConfig(rounds=2)).run(task)     # SAME Phase-0 certify path
#
# Ordering contract the integrator MUST preserve:
#   1. self_test() is called (and must return True) BEFORE adapt()'s Task is fed to the engine.
#      A harness whose self-test has not passed is `harness.trusted == False`; adapt() still
#      builds a Task, but the integrator must not treat its engine result as trusted.
#   2. The harness NEVER computes a promotion-bearing number. self_test() certifies by running
#      the real Phase-0 ResearchEngine, which calls the frozen certifier (vfplatform.sealed /
#      vectorforge.science). The harness only ASSEMBLES the Task + chooses baselines/split/metric.
#   3. baseline_suite() returns seed Programs (recipes); they are the floor/fallback, never the
#      promoter. The engine's SeedProposer already carries the same role; baseline_suite() lets
#      a router pass modality-specific extra seeds into EngineConfig's proposer list if desired.
#   4. split_protocol() returns the (test_frac, val_frac) the harness recommends for its modality;
#      the integrator copies them into EngineConfig (test_frac/val_frac). Defaults match Phase 0.
#
# Why a self-test gate (the load-bearing innovation): a harness is itself untrusted code that
# decides metric, theta, split, and feature layout. A bug there (wrong metric axis, leaky split,
# degenerate adapt) would silently corrupt every downstream certificate. So a harness's numbers
# are trusted ONLY after the harness certifies itself on a dataset whose answer is known to be
# achievable, through the exact same sealed gate the real task will use. If the self-test cannot
# certify the known-good case, the harness is declared untrusted and the integrator declines --
# the same honest-decline discipline the spine uses for tasks.
"""

from __future__ import annotations

import abc
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Repo root on sys.path so sibling frontier modules + the sound certifier resolve when this
# file is imported from anywhere (mirrors certify.py's bootstrap).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.program import Program            # noqa: E402
from frontier.task import Task                  # noqa: E402


@dataclass
class HarnessCertificate:
    """The result of a harness self-test.

    A harness self-test runs the harness's own adapt->split->baseline path on a KNOWN-GOOD
    built-in dataset through the real Phase-0 ResearchEngine, then records whether the frozen
    sealed certifier promoted the winner above the self-test theta. This certificate is what
    licenses trusting the harness's adapt() output downstream.

    Fields:
      harness_key   : the registry key of the harness under test.
      task_kind     : the kind exercised by the self-test ("classification"|"regression").
      dataset       : human name of the built-in known-good dataset.
      metric        : the metric the harness chose for this kind.
      self_theta    : the (deliberately conservative) threshold the self-test must clear.
      certified     : whether the frozen sealed certifier promoted the winner above self_theta.
      sealed_cert   : the verbatim sealed certificate dict from the frozen certifier (or None).
      winner_label  : the label of the winning program (provenance).
      detail         : a short human-readable status / decline reason.
    """

    harness_key: str
    task_kind: str
    dataset: str
    metric: str
    self_theta: float
    certified: bool
    sealed_cert: Optional[dict] = None
    winner_label: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        # So `if cert:` reads as "the harness certified itself".
        return bool(self.certified)


class Harness(abc.ABC):
    """Adapts raw, modality-specific data into a Phase-0 Task and self-certifies first.

    A Harness owns five responsibilities for one family of task types:
      1. adapt(raw...)        -> Task           (data -> rows/metric/theta the spine can run)
      2. baseline_suite(kind) -> [Program]      (the floor recipes; seeds/fallbacks, NOT promoters)
      3. split_protocol(kind) -> (test_frac, val_frac)  (modality-appropriate split fractions)
      4. metric_for(kind)     -> str            (metric on the RIGHT axis for the kind)
      5. self_test()          -> (bool, HarnessCertificate)

    The self-test is the trust boundary: until it passes, `self.trusted` is False and the
    integrator must not promote any downstream number this harness produced.
    """

    #: registry key, e.g. "tabular". Subclasses set this.
    key: str = ""
    #: which task kinds this harness can adapt.
    kinds: Tuple[str, ...] = ()

    def __init__(self):
        self._trusted = False
        self._last_cert: Optional[HarnessCertificate] = None

    # ------------------------------------------------------------------ trust state
    @property
    def trusted(self) -> bool:
        """True only after self_test() has certified this harness on a known-good dataset."""
        return self._trusted

    @property
    def last_certificate(self) -> Optional[HarnessCertificate]:
        """The most recent self-test certificate (None until self_test() runs)."""
        return self._last_cert

    # ------------------------------------------------------------------ adapter API
    @abc.abstractmethod
    def adapt(self, X, y, *, kind: str, theta: float, name: str = "task",
              metric: str = "") -> Task:
        """Turn raw data into a Phase-0 Task (rows/metric/theta) the engine can certify.

        `metric=""` lets the harness pick the right-axis metric for the kind via metric_for().
        The harness chooses NO promotion-bearing number here -- theta is supplied by the caller
        (the goal's verification standard); the harness only assembles the Task.
        """

    @abc.abstractmethod
    def baseline_suite(self, kind: str) -> List[Program]:
        """Return the floor recipes for this kind, as seed Programs.

        These are seeds/baselines/fallbacks (invariant 3): they guarantee a floor and seed the
        search, but they are NEVER the thing that promotes -- the frozen certifier is.
        """

    @abc.abstractmethod
    def metric_for(self, kind: str) -> str:
        """The metric to optimize+certify for this kind, on the correct axis."""

    def split_protocol(self, kind: str) -> Tuple[float, float]:
        """Recommended (test_frac, val_frac) for this modality. Default matches Phase 0."""
        return (0.30, 0.20)

    # ------------------------------------------------------------------ self-test
    @abc.abstractmethod
    def _self_test_case(self) -> Tuple[np.ndarray, np.ndarray, str, str, float]:
        """Provide the KNOWN-GOOD self-test case for this harness.

        Returns (X, y, kind, dataset_name, self_theta). The dataset must be self-contained
        (a sklearn loader or a deterministic synthetic generator) so the self-test needs no
        network and no external files. self_theta must be a threshold that a competent baseline
        is KNOWN to clear on this dataset -- deliberately conservative so the gate tests the
        adapter/metric/split plumbing, not the modeling difficulty.
        """

    def self_test(self, *, rounds: int = 1, seed: int = 0,
                  wall_seconds: float = 45.0, cpu_seconds: int = 40) -> Tuple[bool, HarnessCertificate]:
        """Certify THIS harness on its known-good dataset through the frozen Phase-0 gate.

        We deliberately route the self-test through the real ResearchEngine rather than scoring
        anything ourselves: that exercises the harness's adapt->split->baseline path AND defers
        every number to the audited sealed certifier. If the frozen certifier promotes the
        winner above the conservative self_theta, the harness's plumbing is sound and we set
        `self.trusted = True`. Otherwise the harness stays untrusted and the integrator declines.

        Imported lazily so importing this module never forces engine/sandbox import order issues.
        """
        from frontier.engine import ResearchEngine, EngineConfig

        X, y, kind, dataset, self_theta = self._self_test_case()
        metric = self.metric_for(kind)
        task = self.adapt(X, y, kind=kind, theta=self_theta, name=f"selftest:{self.key}",
                          metric=metric)
        test_frac, val_frac = self.split_protocol(kind)

        # The harness's own baseline suite seeds the search so the self-test exercises the
        # harness's recipes, not just the engine's defaults. We prepend a SeedProposer that
        # serves the harness baselines; the engine's normal proposers still run too.
        from frontier.proposers import SeedProposer, MutationProposer, LLMProposer

        harness_seeds = self.baseline_suite(kind)
        proposers = [_StaticSeedProposer(harness_seeds), SeedProposer(),
                     MutationProposer(), LLMProposer(None)]
        cfg = EngineConfig(rounds=rounds, seed=seed, test_frac=test_frac, val_frac=val_frac,
                           wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
        result = ResearchEngine(cfg, proposers=proposers).run(task)

        cert = HarnessCertificate(
            harness_key=self.key,
            task_kind=kind,
            dataset=dataset,
            metric=metric,
            self_theta=self_theta,
            certified=bool(result.certified),
            sealed_cert=result.certificate,
            winner_label=(result.winner.label if result.winner is not None else ""),
            detail=("self-test certified through frozen sealed gate"
                    if result.certified
                    else f"self-test did NOT certify: {result.decline_reason or 'no winner'}"),
        )
        self._trusted = cert.certified
        self._last_cert = cert
        return cert.certified, cert


class _StaticSeedProposer:
    """Serve a fixed list of harness baseline Programs as round-0 seeds.

    Used inside self_test() so the harness's own baseline_suite() is exercised. It honors the
    engine's `tried_labels` dedup contract (same as SeedProposer) so it does not re-propose a
    label already evaluated in a prior round.
    """

    def __init__(self, programs: List[Program]):
        self._programs = list(programs)

    def propose(self, context: dict) -> List[Program]:
        tried = set(context.get("tried_labels", ()))
        return [p for p in self._programs if p.label not in tried]


class HarnessRegistry:
    """Lookup of harnesses by task-type key (the Phase-3 router's backbone).

    A router maps a problem ontology key (e.g. "tabular", "vision", "text") to the harness that
    knows how to adapt/split/baseline/certify that modality. Registration is explicit; lookup is
    by exact key. A single Harness instance may serve several keys (e.g. one TabularHarness
    registered under both "tabular" and "classification") -- aliasing is supported.

    The registry stores harness INSTANCES (stateful: each remembers its own trust state), so a
    harness self-tested once stays trusted for the life of the process.
    """

    def __init__(self):
        self._by_key: dict = {}

    def register(self, harness: Harness, *keys: str) -> Harness:
        """Register a harness under one or more keys. Defaults to harness.key if no keys given.

        Raises KeyError on a duplicate key to prevent silent shadowing (a router must not have
        two harnesses fighting over "tabular").
        """
        ks = keys or ((harness.key,) if harness.key else ())
        if not ks:
            raise ValueError("cannot register a harness with no key")
        for k in ks:
            if k in self._by_key and self._by_key[k] is not harness:
                raise KeyError(f"harness key {k!r} already registered to {self._by_key[k]!r}")
            self._by_key[k] = harness
        return harness

    def lookup(self, key: str) -> Harness:
        """Return the harness registered for `key`, or raise KeyError listing known keys."""
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"no harness for task-type {key!r}; known: {sorted(self._by_key)}")

    def get(self, key: str) -> Optional[Harness]:
        """Like lookup() but returns None instead of raising (router convenience)."""
        return self._by_key.get(key)

    def keys(self) -> List[str]:
        return sorted(self._by_key)

    def __contains__(self, key: str) -> bool:
        return key in self._by_key


# Module-level default registry the integrator/router shares. tabular.py registers into it.
REGISTRY = HarnessRegistry()


def register(harness: Harness, *keys: str) -> Harness:
    """Register into the shared default REGISTRY (convenience)."""
    return REGISTRY.register(harness, *keys)


def lookup(key: str) -> Harness:
    """Look up in the shared default REGISTRY (convenience)."""
    return REGISTRY.lookup(key)
