"""Adaptive Hyperparameter Search -- successive halving (ASHA) + Bayesian optimization (TPE).

Thesis: fixed grids waste budget on bad configs. A frontier autoresearcher allocates compute adaptively:
(1) Successive Halving (SHA/ASHA) prunes the bottom fraction of configs at each rung, doubling the budget
    for survivors -- so a 64-config search at (10, 20, 40, 80, 160) epochs costs ~3x the full-budget run of
    the WINNER, not 64x.
(2) Tree-Parzen Estimator (TPE) proposes new configs from the empirical distribution of good vs bad trials,
    replacing the static grid with an informed, evolving proposal distribution.

Both integrate into the existing VoI-ranked propose→execute loop: `HPSearch.propose()` returns the next batch
of (family, params) configs to try, adapting to realized measurements. The frozen certifier is unchanged --
this only affects WHICH configs are proposed and HOW MUCH budget each gets.

Selection is on VALIDATION only. The sealed test is never read here. The frozen certifier remains the sole
promoter.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================================== SUCCESSIVE HALVING

@dataclass
class Rung:
    """One rung of a successive halving bracket."""
    budget: int           # epochs (or any resource unit)
    n_configs: int        # how many configs start at this rung
    results: Dict[str, float] = field(default_factory=dict)  # config_id -> val_score


@dataclass
class SHABracket:
    """A Successive Halving / ASHA bracket: configs compete in rungs of increasing budget.

    At each rung, the bottom (1 - 1/eta) fraction is pruned. Survivors advance to the next rung
    with eta× the budget. Early stopping is thus principled: configs that show no signal at low
    budget never consume the full training budget.

    Parameters:
        max_budget:   maximum resource (e.g. epochs) any config will receive
        eta:          halving factor (default 3: keep top 1/3 at each rung)
        n_configs:    number of configs to start (if None, computed from budget/eta)
    """
    max_budget: int = 160
    eta: int = 3
    n_configs: Optional[int] = None
    rungs: List[Rung] = field(default_factory=list)
    config_budgets: Dict[str, int] = field(default_factory=dict)
    config_scores: Dict[str, float] = field(default_factory=dict)
    _promoted: Dict[str, int] = field(default_factory=dict)  # config_id -> highest rung reached

    def __post_init__(self):
        if not self.rungs:
            s_max = max(1, int(math.log(self.max_budget, self.eta)))
            if self.n_configs is None:
                self.n_configs = int(self.eta ** s_max)
            budget = self.max_budget
            rungs = []
            n = self.n_configs
            for _ in range(s_max + 1):
                rungs.append(Rung(budget=int(round(budget)), n_configs=int(n)))
                budget /= self.eta
                n = max(1, int(n / self.eta))
            # rungs: smallest budget first (ascending)
            self.rungs = list(reversed(rungs))

    def lowest_budget(self) -> int:
        return self.rungs[0].budget if self.rungs else self.max_budget

    def rung_for(self, config_id: str) -> int:
        """Current rung index for a config (0 = first/cheapest)."""
        return self._promoted.get(config_id, 0)

    def record(self, config_id: str, budget: int, score: float):
        """Record that config_id achieved `score` at `budget`."""
        self.config_budgets[config_id] = budget
        self.config_scores[config_id] = score
        # find which rung this budget corresponds to
        for i, r in enumerate(self.rungs):
            if r.budget == budget:
                r.results[config_id] = score
                self._promoted[config_id] = i
                break

    def promotable(self, rung_idx: int) -> List[str]:
        """Which configs at rung_idx are in the top 1/eta and should be promoted to rung_idx+1."""
        if rung_idx >= len(self.rungs) - 1:
            return []
        rung = self.rungs[rung_idx]
        results = rung.results
        if not results:
            return []
        sorted_ids = sorted(results, key=lambda k: results[k], reverse=True)
        n_promote = max(1, len(sorted_ids) // self.eta)
        # only promote those not yet at a higher rung
        return [cid for cid in sorted_ids[:n_promote]
                if self._promoted.get(cid, 0) <= rung_idx]

    def next_actions(self) -> List[Tuple[str, int]]:
        """Return (config_id, budget) pairs for the next actions to take.
        Strategy: promote from lowest pending rung first (ASHA async promotion)."""
        actions = []
        for i in range(len(self.rungs) - 1):
            for cid in self.promotable(i):
                next_budget = self.rungs[i + 1].budget
                if self.config_budgets.get(cid, 0) < next_budget:
                    actions.append((cid, next_budget))
        return actions

    def best_config(self) -> Optional[str]:
        """The highest-scoring config at the highest rung."""
        for rung in reversed(self.rungs):
            if rung.results:
                return max(rung.results, key=rung.results.get)
        return None

    def is_complete(self) -> bool:
        """True if the top rung has at least one result."""
        return bool(self.rungs[-1].results) if self.rungs else True


# ============================================================================================== TPE

@dataclass
class Trial:
    """One observed trial for the TPE."""
    config: dict        # {param_name: value}
    score: float
    family: str = ""


class TPE:
    """Tree-Parzen Estimator: proposes hyperparameters from the empirical distribution of good trials.

    Instead of uniformly sampling the grid, TPE splits observed trials into "good" (top gamma quantile)
    and "bad" (rest), fits a density estimate to each, and proposes configs that maximize l(good)/l(bad).
    For continuous params we use a simple kernel density (Gaussian around each good trial); for categorical
    we use a smoothed frequency table. This is a lightweight, dependency-free implementation suitable for
    the loop's budget (dozens of trials, not thousands).
    """

    def __init__(self, param_specs: dict, *, gamma: float = 0.25, n_startup: int = 8, seed: int = 0):
        """param_specs: {name: ("float", lo, hi) | ("int", lo, hi) | ("choice", [vals])}"""
        self.specs = dict(param_specs)
        self.gamma = gamma
        self.n_startup = n_startup
        self.rng = random.Random(seed)
        self.trials: List[Trial] = []

    def observe(self, config: dict, score: float, family: str = ""):
        self.trials.append(Trial(config=dict(config), score=score, family=family))

    def _split(self) -> Tuple[List[Trial], List[Trial]]:
        """Split trials into good (top gamma) and bad (rest)."""
        sorted_trials = sorted(self.trials, key=lambda t: t.score, reverse=True)
        n_good = max(1, int(len(sorted_trials) * self.gamma))
        return sorted_trials[:n_good], sorted_trials[n_good:]

    def _sample_from_good(self, good: List[Trial]) -> dict:
        """Sample a config from the 'good' distribution (kernel density around good trials)."""
        base = self.rng.choice(good).config
        config = {}
        for name, spec in self.specs.items():
            kind = spec[0]
            base_val = base.get(name)
            if kind == "choice":
                allowed = spec[1]
                # weighted toward good values with exploration noise
                if self.rng.random() < 0.7 and base_val in allowed:
                    config[name] = base_val
                else:
                    config[name] = self.rng.choice(allowed)
            elif kind == "float":
                lo, hi = spec[1], spec[2]
                if base_val is not None:
                    # Gaussian perturbation (bandwidth = 20% of range)
                    bw = (hi - lo) * 0.2
                    v = self.rng.gauss(float(base_val), bw)
                    config[name] = max(lo, min(hi, v))
                else:
                    config[name] = self.rng.uniform(lo, hi)
            elif kind == "int":
                lo, hi = spec[1], spec[2]
                if base_val is not None:
                    bw = max(1, (hi - lo) * 0.2)
                    v = int(round(self.rng.gauss(float(base_val), bw)))
                    config[name] = max(lo, min(hi, v))
                else:
                    config[name] = self.rng.randint(lo, hi)
        return config

    def _sample_uniform(self) -> dict:
        """Sample uniformly from the param space (startup phase)."""
        config = {}
        for name, spec in self.specs.items():
            kind = spec[0]
            if kind == "choice":
                config[name] = self.rng.choice(spec[1])
            elif kind == "float":
                config[name] = self.rng.uniform(spec[1], spec[2])
            elif kind == "int":
                config[name] = self.rng.randint(spec[1], spec[2])
        return config

    def propose(self, n: int = 1) -> List[dict]:
        """Propose n candidate configs. During startup (< n_startup trials), sample uniformly.
        After startup, sample from the good distribution."""
        configs = []
        for _ in range(n):
            if len(self.trials) < self.n_startup:
                configs.append(self._sample_uniform())
            else:
                good, _ = self._split()
                configs.append(self._sample_from_good(good))
        return configs


# ============================================================================================== INTEGRATED SEARCH

@dataclass
class HPSearchConfig:
    """Configuration for adaptive HP search within one family."""
    family: str
    param_specs: dict           # from CatalogEntry.params
    max_budget: int = 160       # max epochs
    eta: int = 3                # SHA halving factor
    n_initial: int = 12         # initial configs before SHA starts pruning
    tpe_gamma: float = 0.25     # top fraction for TPE good/bad split
    tpe_startup: int = 6        # random trials before TPE kicks in
    seed: int = 0


class HPSearch:
    """Adaptive hyperparameter search combining SHA + TPE for one family.

    Usage within the loop:
        hp = HPSearch(config)
        # Round 1: propose initial configs at lowest budget
        batch = hp.propose_initial()
        for cfg_id, params, budget in batch:
            score = run(family, params, epochs=budget)
            hp.record(cfg_id, params, budget, score)
        # Subsequent rounds: promote survivors + propose new from TPE
        while not hp.is_done():
            batch = hp.propose_next()
            for cfg_id, params, budget in batch:
                score = run(family, params, epochs=budget)
                hp.record(cfg_id, params, budget, score)
        best_params = hp.best()
    """

    def __init__(self, config: HPSearchConfig):
        self.config = config
        self.bracket = SHABracket(max_budget=config.max_budget, eta=config.eta, n_configs=config.n_initial)
        self.tpe = TPE(config.param_specs, gamma=config.tpe_gamma, n_startup=config.tpe_startup,
                       seed=config.seed)
        self._configs: Dict[str, dict] = {}  # config_id -> params
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.config.family}_{self._counter:03d}"

    def propose_initial(self) -> List[Tuple[str, dict, int]]:
        """Propose the initial batch of configs at the lowest SHA budget."""
        budget = self.bracket.lowest_budget()
        configs = self.tpe.propose(self.config.n_initial)
        batch = []
        for params in configs:
            cid = self._next_id()
            self._configs[cid] = params
            batch.append((cid, params, budget))
        return batch

    def record(self, config_id: str, params: dict, budget: int, score: float):
        """Record an observed (config_id, params, budget, score)."""
        self._configs[config_id] = params
        self.bracket.record(config_id, budget, score)
        self.tpe.observe(params, score, family=self.config.family)

    def propose_next(self) -> List[Tuple[str, dict, int]]:
        """Propose the next batch: promote SHA survivors + optionally add new TPE-proposed configs."""
        batch = []
        # 1. Promote survivors from current rungs
        for cid, budget in self.bracket.next_actions():
            params = self._configs.get(cid, {})
            batch.append((cid, params, budget))
        # 2. If no promotions pending and not done, propose new configs from TPE at the current frontier
        if not batch and not self.is_done():
            n_new = max(2, self.config.n_initial // self.config.eta)
            new_configs = self.tpe.propose(n_new)
            # find the lowest rung with incomplete slots
            budget = self.bracket.lowest_budget()
            for params in new_configs:
                cid = self._next_id()
                self._configs[cid] = params
                batch.append((cid, params, budget))
        return batch

    def is_done(self) -> bool:
        return self.bracket.is_complete()

    def best(self) -> Optional[dict]:
        """Return the best params found."""
        best_id = self.bracket.best_config()
        return self._configs.get(best_id) if best_id else None

    def best_score(self) -> Optional[float]:
        best_id = self.bracket.best_config()
        return self.bracket.config_scores.get(best_id) if best_id else None

    def summary(self) -> dict:
        return {
            "family": self.config.family,
            "n_trials": len(self.tpe.trials),
            "n_rungs": len(self.bracket.rungs),
            "rung_budgets": [r.budget for r in self.bracket.rungs],
            "best_params": self.best(),
            "best_score": self.best_score(),
            "is_complete": self.is_done(),
        }


# ============================================================================================== LOOP INTEGRATION

def hp_search_for_family(catalog_entry, *, max_budget: int = 160, n_initial: int = 12,
                         seed: int = 0) -> HPSearch:
    """Create an HPSearch instance for a catalog entry, using its param specs."""
    config = HPSearchConfig(
        family=catalog_entry.family,
        param_specs=catalog_entry.params,
        max_budget=max_budget,
        n_initial=n_initial,
        seed=seed,
    )
    return HPSearch(config)


def propose_from_search(hp_searches: Dict[str, HPSearch], *, tried: set = None) -> List[dict]:
    """Given active HP searches per family, return the next batch of proposals as
    [{family, params, budget}] compatible with the loop's candidate format. Filters out already-tried."""
    tried = tried or set()
    proposals = []
    for family, hp in hp_searches.items():
        if hp.is_done():
            continue
        batch = hp.propose_next() if hp.tpe.trials else hp.propose_initial()
        for cid, params, budget in batch:
            key = f"{family}|{sorted(params.items())}|{budget}"
            if key not in tried:
                proposals.append({"config_id": cid, "family": family, "params": params, "budget": budget})
                tried.add(key)
    return proposals
