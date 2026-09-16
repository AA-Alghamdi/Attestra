"""THE RESEARCH-RECIPE DSL -- a typed, OPEN training recipe that the regenerative agent mutates.

WHY THIS EXISTS (the anti-menu)
-------------------------------
Until now the autonomous researcher (vfplatform/repr_researcher.py) climbed a hand-authored `REGISTRY`
of ~9 encoders via a 4-rung ladder. That is a MENU: the system can only ever reach what a human enumerated.
This module replaces the encoder-tag-from-a-list with a FULL TRAINING RECIPE whose decisive field --
`backbone` -- is an OPEN string (any timm / Hugging Face model id), and whose `code_patch` field is a slot
for LLM-authored novel transforms/losses/heads (admitted through the frozen authoring sandbox). The reachable
space is therefore "anything published on the Hub / authorable as code", not a fixed catalog.

A Recipe is the unit the generator regenerates and the runner executes:

    backbone        OPEN model id          -- resnet50 / vit_l_16 / timm/vit_large_patch14_dinov2.lvd142m / ...
    adaptation      how the backbone is adapted   -- linear_probe | lora | adapter | vpt | partial_unfreeze | full_ft
    augmentation    input augmentation            -- none | randaug | trivialaug | mixup | cutmix | fixres
    optimizer       -- adamw | sam
    schedule        -- cosine | step | constant      (+ llrd: layer-wise LR decay on/off)
    aggregation     final consolidation           -- single | logit_ensemble | model_soup | distill
    data_strategy   training-data move            -- none | class_rebalance | label_noise_repair | active
    head            head over features for the cheap pilot -- gbm | linear
    code_patch      OPTIONAL LLM-authored source (a featurizer/loss/head), gated by vfplatform.authoring

This intentionally spans the recipe axes a frontier engineer actually moves (backbone, adaptation, aug,
optim, aggregation, data), so changing the recipe can change the BOX, not just hyperparameters inside it.

INVARIANT (the DSL never decides anything)
-------------------------------------------
A Recipe is inert data. It is proposed by the generator, executed by the runner, and PROMOTED only by the
frozen certifier (vfplatform/verification.py Tier 3). Nothing here imports torch, sklearn, or the frozen
science core; the DSL stays pure so it is unit-testable offline and so a malformed recipe can at worst waste
a pilot, never mint a certificate.

CONTRACT: stdlib + vfplatform.regeneration (the typed gene space + operators). `to_genome`/`from_genome`
make a Recipe round-trip through the QD archive and the mutate/recombine/graft operators verbatim.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .regeneration import Genome, Param, Space

# -- the typed, CLOSED choice axes (the OPEN axes are `backbone` and `code_patch`) ----------------------
ADAPTATIONS: Tuple[str, ...] = ("linear_probe", "lora", "adapter", "vpt", "partial_unfreeze", "full_ft")
AUGMENTATIONS: Tuple[str, ...] = ("none", "randaug", "trivialaug", "mixup", "cutmix", "fixres")
OPTIMIZERS: Tuple[str, ...] = ("adamw", "sam")
SCHEDULES: Tuple[str, ...] = ("cosine", "step", "constant")
AGGREGATIONS: Tuple[str, ...] = ("single", "logit_ensemble", "model_soup", "distill")
DATA_STRATEGIES: Tuple[str, ...] = ("none", "class_rebalance", "label_noise_repair", "active")
HEADS: Tuple[str, ...] = ("gbm", "linear", "torch_mlp")  # torch_mlp: the GPU/neural head (gated by the arena)

# Adaptation cost proxy (rung weight): how much GPU a recipe's adaptation move costs, used for the Pareto
# accuracy-vs-cost frontier and for cheap pilot ordering. Frozen-feature probes are cheapest.
ADAPTATION_COST: Dict[str, float] = {
    "linear_probe": 1.0, "lora": 3.0, "adapter": 3.0, "vpt": 2.5,
    "partial_unfreeze": 6.0, "full_ft": 10.0,
}


@dataclass(frozen=True)
class Recipe:
    """A full, executable training recipe. `backbone` is an OPEN model id (not from any list); the other
    axes are typed categorical/numeric. Frozen + hashable so it can key a QD cell and a memoization cache."""
    backbone: str
    adaptation: str = "linear_probe"
    augmentation: str = "none"
    optimizer: str = "adamw"
    schedule: str = "cosine"
    llrd: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    aggregation: str = "single"
    data_strategy: str = "none"
    head: str = "gbm"
    code_patch: Optional[str] = None     # LLM-authored source (admitted via the sandbox) or None
    code_role: Optional[str] = None      # 'featurizer' | 'classifier' | 'regressor' for the patch
    notes: str = ""

    def __post_init__(self):
        if not isinstance(self.backbone, str) or not self.backbone.strip():
            raise ValueError("backbone must be a non-empty model id")
        for fieldname, allowed in (("adaptation", ADAPTATIONS), ("augmentation", AUGMENTATIONS),
                                   ("optimizer", OPTIMIZERS), ("schedule", SCHEDULES),
                                   ("aggregation", AGGREGATIONS), ("data_strategy", DATA_STRATEGIES),
                                   ("head", HEADS)):
            v = getattr(self, fieldname)
            if v not in allowed:
                raise ValueError(f"{fieldname}={v!r} not in {allowed}")

    # -- identity ---------------------------------------------------------------------------------------
    def signature(self) -> str:
        """A stable content hash of the recipe (decisive fields + a hash of any code patch). Two recipes
        with the same signature are the same experiment; used for memoization and the anti-menu novelty
        check. Cosmetic `notes` are excluded."""
        patch_h = hashlib.sha256(self.code_patch.encode()).hexdigest()[:12] if self.code_patch else "-"
        key = (self.backbone, self.adaptation, self.augmentation, self.optimizer, self.schedule,
               int(self.llrd), round(float(self.lr), 6), round(float(self.weight_decay), 6),
               int(self.epochs), self.aggregation, self.data_strategy, self.head, patch_h,
               self.code_role or "-")
        return hashlib.sha256(repr(key).encode()).hexdigest()[:16]

    def label(self) -> str:
        """A short human-readable name for logs/certificates."""
        bits = [self.backbone.split("/")[-1], self.adaptation]
        if self.augmentation != "none":
            bits.append(self.augmentation)
        if self.aggregation != "single":
            bits.append(self.aggregation)
        if self.code_patch:
            bits.append(f"+code[{self.code_role or 'patch'}]")   # the authored estimator -- the head is moot
        else:
            bits.append(f"head={self.head}")                     # the estimator when no authored code
        return " · ".join(bits)

    def cost(self) -> float:
        """A GPU-cost proxy used by the Pareto frontier: adaptation cost scaled by epochs, plus an
        aggregation surcharge (ensembles/soups train multiple members)."""
        agg_mult = {"single": 1.0, "logit_ensemble": 3.0, "model_soup": 3.0, "distill": 2.0}[self.aggregation]
        return ADAPTATION_COST[self.adaptation] * max(1, self.epochs) / 10.0 * agg_mult

    def backbone_family(self) -> str:
        """Coarse family bucket of the backbone (for the QD behavior descriptor). Heuristic on the id
        string; unknown ids bucket as 'other' -- deliberately permissive so a NOVEL backbone is not forced
        into a known cell."""
        b = self.backbone.lower()
        for fam in ("dinov2", "clip", "siglip", "eva02", "eva", "convnext", "swin", "deit", "beit",
                    "vit", "resnet", "efficientnet", "regnet", "mobilenet"):
            if fam in b:
                return "eva02" if fam == "eva" and "eva02" in b else fam
        return "other"

    # -- round-trip with the regeneration gene space (mutate/recombine/graft operate on genomes) --------
    def to_genome(self) -> Genome:
        g: Genome = {
            "backbone": self.backbone, "adaptation": self.adaptation, "augmentation": self.augmentation,
            "optimizer": self.optimizer, "schedule": self.schedule, "llrd": bool(self.llrd),
            "lr": float(self.lr), "weight_decay": float(self.weight_decay), "epochs": int(self.epochs),
            "aggregation": self.aggregation, "data_strategy": self.data_strategy, "head": self.head,
        }
        return g

    @classmethod
    def from_genome(cls, g: Genome, *, code_patch: Optional[str] = None,
                    code_role: Optional[str] = None, notes: str = "") -> "Recipe":
        return cls(
            backbone=str(g["backbone"]), adaptation=str(g["adaptation"]),
            augmentation=str(g.get("augmentation", "none")), optimizer=str(g.get("optimizer", "adamw")),
            schedule=str(g.get("schedule", "cosine")), llrd=bool(g.get("llrd", False)),
            lr=float(g.get("lr", 1e-3)), weight_decay=float(g.get("weight_decay", 1e-4)),
            epochs=int(g.get("epochs", 10)), aggregation=str(g.get("aggregation", "single")),
            data_strategy=str(g.get("data_strategy", "none")), head=str(g.get("head", "gbm")),
            code_patch=code_patch, code_role=code_role, notes=notes)

    def with_(self, **changes) -> "Recipe":
        return replace(self, **changes)


def recipe_space(backbone_pool: Sequence[str]) -> Space:
    """Build the regeneration.Space for the recipe. `backbone_pool` is the CURRENT pool of known backbone
    ids -- it is GROWABLE at runtime (the generator/retrieval adds Hub models to it), so the categorical
    backbone axis is a moving target, NOT a fixed menu. The mutate/recombine operators move among whatever
    is in the pool; novelty (reaching a backbone outside the SEED pool) is tracked separately by the
    DiscoveryLedger. lr/weight_decay are log-scaled numeric axes; epochs is a small integer axis."""
    if not backbone_pool:
        raise ValueError("backbone_pool must be non-empty (seed it with at least one model id)")
    pool = list(dict.fromkeys(backbone_pool))   # de-dup, preserve order
    return Space([
        Param("backbone", choices=pool),
        Param("adaptation", choices=list(ADAPTATIONS)),
        Param("augmentation", choices=list(AUGMENTATIONS)),
        Param("optimizer", choices=list(OPTIMIZERS)),
        Param("schedule", choices=list(SCHEDULES)),
        Param("llrd", choices=[False, True]),
        Param("lr", low=1e-5, high=1e-1),
        Param("weight_decay", low=1e-6, high=1e-2),
        Param("epochs", low=3, high=40, is_int=True),
        Param("aggregation", choices=list(AGGREGATIONS)),
        Param("data_strategy", choices=list(DATA_STRATEGIES)),
        Param("head", choices=list(HEADS)),
    ])


def qd_descriptor(recipe: Recipe) -> Tuple[str, str, str]:
    """MAP-Elites behavior descriptor for a recipe: (backbone_family, adaptation, aggregation). This keeps
    diversity along the axes that change the BOX, so a non-obvious region (e.g. a novel backbone with a
    cheap linear probe) is never crowded out by the currently-dominant one."""
    return (recipe.backbone_family(), recipe.adaptation, recipe.aggregation)


@dataclass
class DiscoveryLedger:
    """The ANTI-MENU accountant. Records the SEED backbones/recipes the system was handed, and every recipe
    the generator actually proposed, so the final certificate can prove the champion was DISCOVERED, not
    enumerated. A run is 'menu-free' iff the certified champion's backbone (or an admitted code_patch) was
    NOT in the seed set -- i.e. the system reached beyond what a human listed.

    This never gates promotion (the frozen certifier does that); it is an honesty instrument that makes the
    'it's not a menu' claim FALSIFIABLE: if the champion is always a seed, the guard says so."""
    seed_backbones: frozenset = field(default_factory=frozenset)
    seed_signatures: frozenset = field(default_factory=frozenset)
    proposed_backbones: set = field(default_factory=set)
    proposed_signatures: set = field(default_factory=set)
    retrieved_backbones: set = field(default_factory=set)   # backbones added at runtime from Hub/literature
    literature_backbones: set = field(default_factory=set)  # backbones surfaced by LITERATURE retrieval
    literature_provenance: dict = field(default_factory=dict)  # ingredient id/name -> source record

    @classmethod
    def from_seeds(cls, seed_recipes: Sequence[Recipe]) -> "DiscoveryLedger":
        return cls(seed_backbones=frozenset(r.backbone for r in seed_recipes),
                   seed_signatures=frozenset(r.signature() for r in seed_recipes))

    def record_proposal(self, recipe: Recipe, *, retrieved: bool = False) -> None:
        self.proposed_backbones.add(recipe.backbone)
        self.proposed_signatures.add(recipe.signature())
        if retrieved and recipe.backbone not in self.seed_backbones:
            self.retrieved_backbones.add(recipe.backbone)

    def novelty(self, champion: Recipe) -> dict:
        """Classify how far beyond the seed menu the champion is."""
        b_novel = champion.backbone not in self.seed_backbones
        sig_novel = champion.signature() not in self.seed_signatures
        code_novel = champion.code_patch is not None
        lit_grounded = champion.backbone in self.literature_backbones
        prov = self.literature_provenance.get(champion.backbone) if lit_grounded else None
        return {
            "champion": champion.label(),
            "champion_backbone": champion.backbone,
            "backbone_is_novel": bool(b_novel),
            "recipe_is_novel": bool(sig_novel),
            "uses_authored_code": bool(code_novel),
            "from_retrieval": champion.backbone in self.retrieved_backbones,
            # did the champion's backbone come from reading the LITERATURE (a real paper/repo/Hub hit)?
            "literature_grounded": bool(lit_grounded),
            "literature_source": prov,
            # the headline boolean: did the system reach BEYOND the seed menu to win?
            "menu_free": bool(b_novel or code_novel),
            "n_seed_backbones": len(self.seed_backbones),
            "n_proposed_backbones": len(self.proposed_backbones),
            "n_proposed_recipes": len(self.proposed_signatures),
            "n_retrieved_backbones": len(self.retrieved_backbones),
            "n_literature_backbones": len(self.literature_backbones),
        }


__all__ = [
    "Recipe", "recipe_space", "qd_descriptor", "DiscoveryLedger",
    "ADAPTATIONS", "AUGMENTATIONS", "OPTIMIZERS", "SCHEDULES", "AGGREGATIONS", "DATA_STRATEGIES",
    "HEADS", "ADAPTATION_COST",
]
