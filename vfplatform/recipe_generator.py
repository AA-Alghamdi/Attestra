"""THE REGENERATIVE GENERATOR -- proposes & mutates FULL recipes; this is what replaces the menu.

WHY THIS EXISTS
---------------
The old proposer (`ReprResearcher._propose`) enumerated a fixed encoder REGISTRY. This generator instead
REGENERATES recipes over the open space (vfplatform/recipe.py): it draws backbones from the published zoo
(vfplatform/model_fetcher.py), applies the evolutionary operators (vfplatform/regeneration.py:
mutate / recombine / graft), grafts known-good motifs from a LITERATURE library, and -- when an authorer is
supplied -- attaches LLM-authored novel code patches (admitted through the frozen authoring sandbox). It is
the `expand()` of a search.SearchProblem, so vfplatform/search.py's best-first recursion + plateau
escalation drive it directly.

The four move classes (escalate.py) change the KIND of regeneration, mirroring how an engineer escalates:
  * model            -> swap the BACKBONE (the decisive axis), sampled from the open pool + fresh discovery
  * features         -> vary augmentation / head / attach an authored featurizer patch
  * capacity         -> escalate adaptation (linear->lora->...->full_ft), epochs, and aggregation (soup/ensemble)
  * data_acquisition -> vary the data strategy (rebalance / label-noise repair / active)

INVARIANT
---------
The generator only PROPOSES. It never evaluates and never promotes; fitness comes from the runner and
promotion only from the frozen Tier-3 certifier. Deterministic given a seed. Every proposal is logged in the
DiscoveryLedger so the anti-menu claim stays falsifiable.

CONTRACT: vfplatform.recipe / .regeneration / .model_fetcher (+ an OPTIONAL injected code authorer and an
OPTIONAL injected discovery fn -- both default to safe, offline behavior).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from . import model_fetcher as MF
from . import regeneration as RG
from .recipe import (ADAPTATIONS, AGGREGATIONS, AUGMENTATIONS, DATA_STRATEGIES, DiscoveryLedger, Recipe,
                     qd_descriptor, recipe_space)

# An ordered escalation of adaptation strength (how the capacity move climbs).
_ADAPT_LADDER = ["linear_probe", "lora", "adapter", "vpt", "partial_unfreeze", "full_ft"]
# An ordered escalation of aggregation (consolidation strength).
_AGG_LADDER = ["single", "logit_ensemble", "model_soup", "distill"]

# A callable that, given (code_role, task_shape), returns admitted source or None. Injected (LLM-backed).
CodeAuthorer = Callable[[str, str], Optional[str]]
# A callable that returns fresh backbone ids to grow the pool (injected; defaults to model_fetcher).
DiscoverFn = Callable[[], Sequence[str]]


def literature_library() -> RG.LiteratureLibrary:
    """A small library of PARTIAL recipe motifs distilled from the transfer / distribution-shift literature.
    These are grafted (not enumerated as complete recipes) so search is biased toward published ideas while
    the full recipe around them is still regenerated. Each motif is a partial genome (only the genes it
    constrains); the rest is inherited from the parent and clamped to the space."""
    return RG.LiteratureLibrary([
        # fine-grained transfer: a strong self-supervised ViT + a cheap probe is the textbook lever
        RG.Motif("ssl_linear_probe", {"adaptation": "linear_probe", "head": "gbm"}),
        # parameter-efficient adaptation when full-FT is wasteful (LoRA / adapters / VPT)
        RG.Motif("peft_lora", {"adaptation": "lora", "schedule": "cosine", "llrd": True}),
        RG.Motif("visual_prompt", {"adaptation": "vpt", "schedule": "cosine"}),
        # FixRes-style: stronger augmentation + longer schedule for resolution/robustness
        RG.Motif("fixres_randaug", {"augmentation": "randaug", "epochs": 25, "schedule": "cosine"}),
        RG.Motif("mixup_robust", {"augmentation": "mixup", "optimizer": "adamw", "weight_decay": 5e-2}),
        # consolidation: model soups / logit ensembles for the final champion
        RG.Motif("model_soup", {"aggregation": "model_soup"}),
        RG.Motif("logit_ensemble", {"aggregation": "logit_ensemble"}),
        # distribution-shift data moves
        RG.Motif("rebalance", {"data_strategy": "class_rebalance"}),
        RG.Motif("noise_repair", {"adaptation": "linear_probe", "data_strategy": "label_noise_repair"}),
    ])


@dataclass
class GeneratorConfig:
    task_hint: str = "general"
    fanout: int = 6              # children proposed per expand()
    pool_limit: int = 24         # cap on the offline backbone pool
    online_discovery: bool = False
    graft_prob: float = 0.35     # chance a child is grafted with a literature motif
    code_prob: float = 0.0       # chance a 'features' child gets an authored code patch (needs authorer)
    code_roles: tuple = ("featurizer",)  # which authored code roles to propose on a 'features' move
    gpu_heads: bool = False      # also propose head='torch_mlp' on a 'features' move (gated by the arena)


class RecipeGenerator:
    """Open-ended recipe proposer. Holds the growable backbone pool + the evolutionary operators + the
    literature library, and exposes `expand(recipe, move_class)` for the search."""

    def __init__(self, seed_recipes: Sequence[Recipe], *, config: Optional[GeneratorConfig] = None,
                 seed: int = 0, code_authorer: Optional[CodeAuthorer] = None,
                 discover_fn: Optional[DiscoverFn] = None, task_shape: str = "binary",
                 scout: Optional["object"] = None):
        if not seed_recipes:
            raise ValueError("need at least one seed recipe (the weak champion to start from)")
        self.config = config or GeneratorConfig()
        self.rng = random.Random(seed)
        self.task_shape = task_shape
        self.code_authorer = code_authorer
        self.ledger = DiscoveryLedger.from_seeds(seed_recipes)

        # LITERATURE GROUNDING (the deepest anti-menu): when a LiteratureScout is supplied, its RETRIEVED
        # motifs replace the hard-coded library and its RETRIEVED backbone ids are unioned into discovery,
        # each tagged with provenance so the champion can be traced to a real paper/repo/Hub hit.
        self.scout = scout
        self._lit_ids: set = set()
        if scout is not None:
            self._lit_ids = set(scout.backbones())
            self.ledger.literature_provenance.update(getattr(scout, "provenance", {}) or {})
            lib = scout.library()
            self.library = lib if lib.motifs else literature_library()
        else:
            self.library = literature_library()

        seed_backbones = [r.backbone for r in seed_recipes]
        zoo_fn = discover_fn or (lambda: MF.discover_backbones(
            self.config.task_hint, limit=self.config.pool_limit, seed_pool=seed_backbones,
            online=self.config.online_discovery))
        if scout is not None:
            # literature ids first (so they enter the pool even under a tight pool cap), then the open zoo
            self._discover_fn = lambda: list(scout.backbones()) + list(zoo_fn())
        else:
            self._discover_fn = zoo_fn
        # the OPEN, growable pool: seeds + whatever discovery returns
        self.pool: List[str] = list(dict.fromkeys(seed_backbones))
        self.grow_pool()
        self.space = recipe_space(self.pool)
        for r in seed_recipes:
            self.ledger.record_proposal(r)

    # -- the open pool (anti-menu) ----------------------------------------------------------------------
    def grow_pool(self) -> int:
        """Pull fresh backbone ids from discovery into the pool. Returns how many NEW ids were added. This
        is the mechanism that makes the backbone axis open: the pool grows beyond the seeds at runtime."""
        added = 0
        for mid in self._discover_fn():
            if mid not in self.pool:
                self.pool.append(mid)
                self.ledger.retrieved_backbones.add(mid)
                if mid in self._lit_ids:
                    self.ledger.literature_backbones.add(mid)
                added += 1
        if added:
            self.space = recipe_space(self.pool)
        return added

    # -- the SearchProblem.expand contract --------------------------------------------------------------
    def expand(self, recipe: Recipe, move_class: str) -> List[Recipe]:
        """Regenerate children of `recipe` under the given move class. Open-ended: 'model' draws from the
        whole pool (and grows it on demand); 'features'/'capacity'/'data_acquisition' escalate the
        corresponding axes via the evolutionary operators + literature grafts."""
        if move_class == "model":
            children = self._expand_model(recipe)
        elif move_class == "features":
            children = self._expand_features(recipe)
        elif move_class == "capacity":
            children = self._expand_capacity(recipe)
        elif move_class in ("data_acquisition", "data"):
            children = self._expand_data(recipe)
        else:
            children = self._expand_model(recipe)

        # de-dup by signature against the parent; record proposals for the anti-menu ledger
        seen = {recipe.signature()}
        out: List[Recipe] = []
        for c in children:
            s = c.signature()
            if s in seen:
                continue
            seen.add(s)
            out.append(c)
            self.ledger.record_proposal(c, retrieved=c.backbone in self.ledger.retrieved_backbones)
        return out

    def _maybe_graft(self, recipe: Recipe) -> Recipe:
        if self.rng.random() < self.config.graft_prob and self.library.motifs:
            g = self.library.graft(recipe.to_genome(), self.rng, self.space)
            return Recipe.from_genome(g, notes="graft")
        return recipe

    def _expand_model(self, recipe: Recipe) -> List[Recipe]:
        # if the pool is small relative to fanout, grow it from discovery (open-ended)
        if len(self.pool) < self.config.fanout * 3:
            self.grow_pool()
        others = [b for b in self.pool if b != recipe.backbone]
        self.rng.shuffle(others)
        out: List[Recipe] = []
        for b in others[: self.config.fanout]:
            child = recipe.with_(backbone=b, notes="model-swap")
            out.append(self._maybe_graft(child))
        return out

    def _expand_features(self, recipe: Recipe) -> List[Recipe]:
        base: List[Recipe] = []
        for aug in AUGMENTATIONS:
            if aug != recipe.augmentation:
                base.append(recipe.with_(augmentation=aug, notes="aug"))
        base.append(recipe.with_(head="linear" if recipe.head == "gbm" else "gbm", notes="head"))
        # the GPU/neural head: an explicit candidate when the arena can execute it (allow_torch_head). On a
        # GPU box this fit runs on cuda; on CPU it device-swaps. Only proposed off a non-neural head so the
        # menu stays the linear/gbm/torch_mlp triad rather than churning torch_mlp -> torch_mlp.
        if self.config.gpu_heads and recipe.head != "torch_mlp":
            base.append(recipe.with_(head="torch_mlp", notes="gpu-head"))
        self.rng.shuffle(base)
        # optionally attach LLM/template-authored code patches (open-ended code generation). Each role
        # (featurizer / classifier / ...) is authored as NOVEL source and admitted through the frozen sandbox
        # before it can spend a peek; the arena EXECUTES it and the frozen certifier judges it. Authored
        # patches are the open-ended lever, so they are ALWAYS kept (never shuffled out of the fanout).
        code: List[Recipe] = []
        if self.code_authorer is not None and self.rng.random() < self.config.code_prob:
            for role in self.config.code_roles:
                src = self.code_authorer(role, self.task_shape)
                if src:
                    code.append(recipe.with_(code_patch=src, code_role=role, notes=f"authored-{role}"))
        return (code + base)[: max(self.config.fanout, len(code))]

    def _expand_capacity(self, recipe: Recipe) -> List[Recipe]:
        out: List[Recipe] = []
        i = _ADAPT_LADDER.index(recipe.adaptation) if recipe.adaptation in _ADAPT_LADDER else 0
        if i + 1 < len(_ADAPT_LADDER):
            out.append(recipe.with_(adaptation=_ADAPT_LADDER[i + 1], notes="adapt-up"))
        j = _AGG_LADDER.index(recipe.aggregation) if recipe.aggregation in _AGG_LADDER else 0
        if j + 1 < len(_AGG_LADDER):
            out.append(recipe.with_(aggregation=_AGG_LADDER[j + 1], notes="agg-up"))
        out.append(recipe.with_(epochs=min(40, int(recipe.epochs * 2)), notes="epochs-up"))
        out.append(recipe.with_(llrd=not recipe.llrd, notes="llrd-toggle"))
        self.rng.shuffle(out)
        return out[: self.config.fanout]

    def _expand_data(self, recipe: Recipe) -> List[Recipe]:
        out = [recipe.with_(data_strategy=ds, notes="data") for ds in DATA_STRATEGIES
               if ds != recipe.data_strategy]
        self.rng.shuffle(out)
        return out[: self.config.fanout]

    # -- a free-form regeneration step (used by the QD archive driver) ----------------------------------
    def regenerate_child(self, parents: Sequence[Recipe]) -> Recipe:
        """One QD-style regeneration step from one or two parents: recombine then mutate, with a chance of a
        literature graft. This is the population operator (vfplatform/regeneration) applied to recipes."""
        if len(parents) >= 2 and self.rng.random() < 0.5:
            g = RG.recombine(parents[0].to_genome(), parents[1].to_genome(), self.rng)
        else:
            g = RG.mutate(parents[0].to_genome(), self.space, self.rng, rate=0.34)
        child = Recipe.from_genome(g, notes="regenerate")
        child = self._maybe_graft(child)
        self.ledger.record_proposal(child, retrieved=child.backbone in self.ledger.retrieved_backbones)
        return child

    def descriptor(self, recipe: Recipe):
        return qd_descriptor(recipe)


__all__ = ["RecipeGenerator", "GeneratorConfig", "literature_library", "CodeAuthorer", "DiscoverFn"]
