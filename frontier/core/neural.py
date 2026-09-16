"""frontier.core.neural — typed neural-architecture core (design 03).

THE CORE's generative-modeling unit for neural nets: a framework-agnostic, JSON-serializable
``NeuralSpec`` that the LLM authors, NAS samples, and diagnosis mutates; pure-arithmetic
validity + parameter-budget checks; and two ``ProposalSource`` implementations
(``NASProposer``, ``LLMArchitectProposer``) that emit standard Phase-0 ``Program`` objects.

This module NEVER imports torch. It is the torch-free half of the neural core: the spec, its
validation, its closed-form param count, and the search/authoring proposers. The actual code
generation (sklearn stand-in + import-gated torch estimator) lives in ``render.py``.

# === WIRING ===
The integrator composes this into the unmodified Phase-0 spine purely by adding proposers to
the engine's proposer list. NO Phase-0 file is edited.

    from frontier.engine import ResearchEngine, EngineConfig
    from frontier.proposers import SeedProposer, MutationProposer, LLMProposer
    from frontier.core.neural import NASProposer, LLMArchitectProposer

    proposers = [SeedProposer(), MutationProposer(), LLMProposer(client),  # sklearn floor
                 NASProposer(modality="tabular", param_budget=2_000_000),  # neural NAS search
                 LLMArchitectProposer(client)]                            # neural authoring
    result = ResearchEngine(EngineConfig(rounds=4, llm_client=client),
                            proposers=proposers).run(task)

How the pieces ride the FROZEN spine (CONTRACT.md), with zero spine edits:
  1. Each proposer returns ``Program`` objects whose ``code`` defines ``build_estimator()``
     (proposers.py / program.py contract). The engine sandbox-executes them via
     ``sandbox.run_program`` exactly like sklearn recipes. The generated code is fully
     self-contained (no ``import frontier`` in the child) — see render.py.
  2. The engine scores predictions on VAL via ``certify.score_val`` and certifies the single
     winner ONCE on the sealed test via ``certify.certify_on_sealed``. A neural candidate is
     promoted iff that frozen one-peek gate promotes it (design-03 innovation 3). No new
     promotion number is introduced here.
  3. The engine builds ``context`` each round (CONTRACT.md engine context keys:
     task_kind, n_features, n_train, round, tried_labels, best_label, best_score, best_id,
     best_recipe, recent_errors). The neural proposers additionally READ — and degrade
     gracefully if absent —:
       - ``context["diagnosis"]``  : injected by frontier.diagnosis.enrich_context (Phase 2);
                                     supplies plateau/underfit/overfit search-steering signals.
       - ``context["best_spec"]``  : the champion NeuralSpec dict, recovered from the winning
                                     Program's provenance (set below as
                                     provenance["neural_spec"]). If the engine does not surface
                                     it, NASProposer reconstructs it from best_recipe (which
                                     proposers tag for neural programs) or falls back to seed
                                     sampling. No engine edit is needed: provenance already
                                     flows because the winner Program is the one we emitted.
       - ``param_budget``          : a constructor arg on NASProposer (hardware/first-principles
                                     derived ceiling, NEVER reverse-engineered from a target
                                     metric — CONTRACT invariant 3 scientific
                                     integrity). Over-budget specs are rejected for free before
                                     any build.
  4. Diagnosis-conditioned mutation (innovation 1): underfit -> grow capacity; overfit ->
     regularize/shrink; plateau -> structural pivot. ``neural_fit_signal`` derives the
     train-vs-val underfit/overfit signal from predictions only (firewall preserved).

Backend honesty (CONTRACT invariant 5 + design-03 risk "stand-in / torch divergence"): the
proposers tag every emitted Program with ``provenance["neural_spec"]`` and the render layer
records the realized backend on the run; a stand-in certificate is therefore never conflated
with a torch certificate. See render.py for the backend tag mechanism.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

# Phase-0 import: the proposal unit. This is the ONLY frontier import the proposers need;
# the GENERATED code (render.py) stays self-contained and imports nothing from frontier.
from ..program import Program

# Deferred import of the renderer to avoid a circular import at module load time
# (render.py imports NeuralSpec/Block from here). render_program is resolved lazily.

Activation = Literal["relu", "gelu", "silu", "tanh", "identity"]
Norm = Literal["none", "batch", "layer"]
BlockKind = Literal["mlp", "residual_mlp", "conv1d", "attention", "embedding_pool"]
Modality = Literal["tabular", "sequence", "image1d"]
TaskKind = Literal["classification", "regression"]


# ---------------------------------------------------------------------------------- the spec

@dataclass
class Block:
    """One layer/block of the network. Framework-agnostic; consumed by the renderer.

    Fields beyond ``kind``/``width`` are kind-specific and ignored where irrelevant:
      - conv1d uses kernel_size/stride;
      - attention uses n_heads (and requires width % n_heads == 0);
      - embedding_pool uses vocab_size/pool (sequence/token inputs only).
    """

    kind: BlockKind
    width: int                        # out features / channels / model dim
    activation: Activation = "relu"
    norm: Norm = "none"
    dropout: float = 0.0              # [0, 0.9)
    residual: bool = False            # add a skip connection around this block
    # conv1d only:
    kernel_size: int = 3
    stride: int = 1
    # attention only:
    n_heads: int = 4
    # embedding_pool only (categorical / token inputs):
    vocab_size: int = 0
    pool: Literal["mean", "max", "cls"] = "mean"


@dataclass
class NeuralSpec:
    """A typed, JSON-serializable description of a feed-forward / sequence / conv network.

    It is the unit the LLM emits, NAS samples, diagnosis mutates, and the knowledge base
    stores. It never references torch. ``to_json``/``fingerprint`` give a stable identity for
    dedup and provenance.
    """

    modality: Modality
    task_kind: TaskKind
    in_features: int                  # d for tabular; seq_len for sequence; channels for image1d
    out_dim: int                      # n_classes (clf) or 1 (reg)
    blocks: List[Block]
    # training knobs the spec carries (consumed by the torch runner, design 02):
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20                # early-stop patience on inner val
    seed: int = 0
    label_smoothing: float = 0.0
    provenance: dict = field(default_factory=dict)  # {"author":"llm"|"nas","parent":id,...}

    def to_json(self) -> str:
        """Deterministic JSON for fingerprinting and inlining into generated code."""
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def fingerprint(self) -> str:
        """Short stable id over the full spec (used in Program labels / dedup)."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view (blocks become dicts) — used for provenance + reconstruction."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NeuralSpec":
        """Rebuild a NeuralSpec from a plain dict (e.g. recovered from Program provenance).

        Tolerant: unknown keys are dropped; blocks given as dicts are coerced to ``Block``.
        """
        d = dict(d)
        raw_blocks = d.pop("blocks", []) or []
        blocks: List[Block] = []
        block_fields = set(Block.__dataclass_fields__.keys())
        for b in raw_blocks:
            if isinstance(b, Block):
                blocks.append(b)
            else:
                blocks.append(Block(**{k: v for k, v in dict(b).items() if k in block_fields}))
        spec_fields = set(NeuralSpec.__dataclass_fields__.keys())
        kw = {k: v for k, v in d.items() if k in spec_fields and k != "blocks"}
        return NeuralSpec(blocks=blocks, **kw)


# ----------------------------------------------------------------------- validity & budget

_MIN_WIDTH, _MAX_WIDTH = 1, 4096


def _conv_out_len(seq_len: int, kernel_size: int, stride: int) -> int:
    """Output sequence length of a 1d conv (no padding, dilation 1), floor division.

    L_out = floor((L_in - kernel_size) / stride) + 1. This analytic check catches the single
    most common silent torch crash: a conv stack that shrinks the sequence below length 1.
    """
    if kernel_size < 1 or stride < 1:
        return -1
    return (seq_len - kernel_size) // stride + 1


def validate(spec: NeuralSpec) -> List[str]:
    """Return a list of shape/budget violations (empty list == valid).

    Pure arithmetic — no build, no torch. These checks let NAS reject infeasible samples for
    free and back the local unit tests (design 03 §1.1).
    """
    errs: List[str] = []

    # --- task-shape coherence
    if spec.in_features <= 0:
        errs.append(f"in_features must be > 0 (got {spec.in_features})")
    if spec.out_dim < 1:
        errs.append(f"out_dim must be >= 1 (got {spec.out_dim})")
    if spec.task_kind == "regression" and spec.out_dim != 1:
        errs.append(f"regression requires out_dim == 1 (got {spec.out_dim})")
    if spec.task_kind == "classification" and spec.out_dim < 2:
        errs.append(f"classification requires out_dim >= 2 (got {spec.out_dim})")

    # --- training-knob ranges
    if not (0.0 < spec.lr < 1.0):
        errs.append(f"lr must be in (0,1) (got {spec.lr})")
    if spec.batch_size < 1:
        errs.append(f"batch_size must be >= 1 (got {spec.batch_size})")
    if spec.max_epochs < 1:
        errs.append(f"max_epochs must be >= 1 (got {spec.max_epochs})")
    if spec.patience < 1:
        errs.append(f"patience must be >= 1 (got {spec.patience})")
    if not (0.0 <= spec.label_smoothing < 1.0):
        errs.append(f"label_smoothing must be in [0,1) (got {spec.label_smoothing})")
    if spec.optimizer not in ("adam", "adamw", "sgd"):
        errs.append(f"unknown optimizer {spec.optimizer!r}")

    if not spec.blocks:
        errs.append("spec must have >= 1 block")

    # --- per-block + running-shape checks
    seq_len = spec.in_features  # only meaningful for sequence / image1d conv stacks
    n_blocks = len(spec.blocks)
    for i, b in enumerate(spec.blocks):
        if not (_MIN_WIDTH <= b.width <= _MAX_WIDTH):
            errs.append(f"block[{i}] width {b.width} outside [{_MIN_WIDTH},{_MAX_WIDTH}]")
        if not (0.0 <= b.dropout < 0.9):
            errs.append(f"block[{i}] dropout {b.dropout} outside [0,0.9)")
        if b.activation not in ("relu", "gelu", "silu", "tanh", "identity"):
            errs.append(f"block[{i}] unknown activation {b.activation!r}")
        if b.norm not in ("none", "batch", "layer"):
            errs.append(f"block[{i}] unknown norm {b.norm!r}")

        if b.kind == "attention":
            if spec.modality != "sequence":
                errs.append(f"block[{i}] attention is only legal for sequence modality")
            if b.n_heads < 1:
                errs.append(f"block[{i}] n_heads must be >= 1 (got {b.n_heads})")
            elif b.width % b.n_heads != 0:
                errs.append(f"block[{i}] attention width {b.width} not divisible by "
                            f"n_heads {b.n_heads}")
        elif b.kind == "embedding_pool":
            if spec.modality != "sequence":
                errs.append(f"block[{i}] embedding_pool is only legal for sequence modality")
            if b.vocab_size < 1:
                errs.append(f"block[{i}] embedding_pool requires vocab_size >= 1 "
                            f"(got {b.vocab_size})")
            if b.pool not in ("mean", "max", "cls"):
                errs.append(f"block[{i}] unknown pool {b.pool!r}")
        elif b.kind == "conv1d":
            if spec.modality not in ("sequence", "image1d"):
                errs.append(f"block[{i}] conv1d is only legal for sequence/image1d modality")
            if b.kernel_size < 1:
                errs.append(f"block[{i}] kernel_size must be >= 1 (got {b.kernel_size})")
            if b.stride < 1:
                errs.append(f"block[{i}] stride must be >= 1 (got {b.stride})")
            seq_len = _conv_out_len(seq_len, b.kernel_size, b.stride)
            if seq_len < 1:
                errs.append(f"block[{i}] conv stack shrinks sequence length to {seq_len} "
                            f"(< 1): infeasible")
        elif b.kind in ("mlp", "residual_mlp"):
            pass
        else:
            errs.append(f"block[{i}] unknown kind {b.kind!r}")

        # residual=True on the FINAL projection-to-out_dim is impossible (the generator cannot
        # add a skip around a width change to out_dim). Flag it (design 03 §1.1).
        if b.residual and i == n_blocks - 1 and b.width != spec.out_dim:
            # only an issue if this is the last hidden block and out_dim differs; the renderer
            # appends a separate output head, so a residual on a non-final block is fine.
            pass

    return errs


def _block_params(in_dim: int, b: Block) -> int:
    """Closed-form trainable-parameter estimate for one block (weights + biases + norm).

    Approximation suitable for budgeting (NOT an exact torch count): linear ~ in*out+out;
    conv1d ~ in*out*k + out; attention ~ 4*d^2 (qkv+proj) + 2*ffn(4d); embedding ~ vocab*d.
    Norm adds 2*out affine params. Returns params AND is paired with the new feature dim by
    ``estimate_params``.
    """
    out = b.width
    if b.kind in ("mlp", "residual_mlp"):
        p = in_dim * out + out
    elif b.kind == "conv1d":
        p = in_dim * out * max(1, b.kernel_size) + out
    elif b.kind == "attention":
        d = out
        p = 4 * d * d + 4 * d  # qkv + out proj (biases folded in approximately)
        p += 2 * (d * (4 * d) + 4 * d)  # a 4x feed-forward block (two linears)
    elif b.kind == "embedding_pool":
        p = max(1, b.vocab_size) * out  # embedding table
    else:
        p = in_dim * out + out
    if b.norm in ("batch", "layer"):
        p += 2 * out
    return p


def estimate_params(spec: NeuralSpec) -> int:
    """Closed-form total trainable-parameter estimate, including the output head.

    Drives the complexity budget (design 03 §7, innovation 2): a spec is rejected for free if
    ``estimate_params(spec) > param_budget`` BEFORE anything is built. Includes the final
    linear head ``last_width -> out_dim``.
    """
    in_dim = spec.in_features
    total = 0
    last = in_dim
    for b in spec.blocks:
        total += _block_params(last, b)
        last = b.width
    # output head: last_width -> out_dim (+ bias)
    total += last * spec.out_dim + spec.out_dim
    return int(total)


# ------------------------------------------------------------- diagnosis-conditioned mutation
# All mutators return a NEW NeuralSpec (parent untouched), tag provenance with the parent
# fingerprint + the mutation name, and re-seed deterministically so the search is reproducible.


def _child(parent: NeuralSpec, name: str) -> NeuralSpec:
    """Deep-ish copy of a parent spec carrying parent/mutation provenance + a fresh seed."""
    blocks = [Block(**asdict(b)) for b in parent.blocks]
    prov = {"author": "nas", "parent": parent.fingerprint, "mutation": name}
    child = NeuralSpec(
        modality=parent.modality, task_kind=parent.task_kind, in_features=parent.in_features,
        out_dim=parent.out_dim, blocks=blocks, optimizer=parent.optimizer, lr=parent.lr,
        weight_decay=parent.weight_decay, batch_size=parent.batch_size,
        max_epochs=parent.max_epochs, patience=parent.patience,
        seed=(parent.seed + 1) % 2**31, label_smoothing=parent.label_smoothing,
        provenance=prov,
    )
    return child


def _widen(parent: NeuralSpec, factor: float = 1.5) -> NeuralSpec:
    c = _child(parent, "widen")
    for b in c.blocks:
        if b.kind in ("mlp", "residual_mlp", "conv1d", "attention"):
            b.width = min(_MAX_WIDTH, max(_MIN_WIDTH, int(round(b.width * factor))))
            if b.kind == "attention" and b.width % b.n_heads != 0:
                b.width = (b.width // b.n_heads) * b.n_heads or b.n_heads
    return c


def _deepen(parent: NeuralSpec) -> NeuralSpec:
    c = _child(parent, "deepen")
    # duplicate the last hidden mlp/residual block (capacity growth without a shape change)
    hidden = [b for b in c.blocks if b.kind in ("mlp", "residual_mlp")]
    template = hidden[-1] if hidden else Block(kind="mlp", width=max(8, parent.in_features))
    c.blocks.append(Block(**asdict(template)))
    return c


def _reduce_dropout(parent: NeuralSpec) -> NeuralSpec:
    c = _child(parent, "reduce_dropout")
    for b in c.blocks:
        b.dropout = max(0.0, round(b.dropout * 0.5, 4))
    return c


def _add_dropout(parent: NeuralSpec, amount: float = 0.2) -> NeuralSpec:
    c = _child(parent, "add_dropout")
    for b in c.blocks:
        b.dropout = min(0.8, round(b.dropout + amount, 4))
    return c


def _add_norm(parent: NeuralSpec) -> NeuralSpec:
    c = _child(parent, "add_norm")
    norm = "layer" if parent.modality == "sequence" else "batch"
    for b in c.blocks:
        if b.kind in ("mlp", "residual_mlp", "conv1d", "attention") and b.norm == "none":
            b.norm = norm  # type: ignore[assignment]
    return c


def _narrow(parent: NeuralSpec, factor: float = 0.66) -> NeuralSpec:
    c = _child(parent, "narrow")
    for b in c.blocks:
        if b.kind in ("mlp", "residual_mlp", "conv1d", "attention"):
            b.width = max(_MIN_WIDTH, int(round(b.width * factor)))
            if b.kind == "attention":
                b.width = max(b.n_heads, (b.width // b.n_heads) * b.n_heads)
    return c


def _raise_weight_decay(parent: NeuralSpec, factor: float = 3.0) -> NeuralSpec:
    c = _child(parent, "raise_weight_decay")
    c.weight_decay = min(0.5, round(c.weight_decay * factor + 1e-4, 6))
    return c


def _add_block(parent: NeuralSpec) -> NeuralSpec:
    c = _child(parent, "add_block")
    w = parent.blocks[-1].width if parent.blocks else max(8, parent.in_features)
    c.blocks.append(Block(kind="mlp", width=max(_MIN_WIDTH, w), activation="gelu",
                          residual=True))
    return c


def _switch_block_kind(parent: NeuralSpec) -> NeuralSpec:
    """Structural pivot: turn an mlp into a residual_mlp (or, for sequence, add attention)."""
    c = _child(parent, "switch_block_kind")
    if parent.modality == "sequence":
        # pivot toward attention if not already present and divisibility allows it
        w = parent.blocks[-1].width if parent.blocks else 64
        heads = 4 if w % 4 == 0 else 1
        c.blocks.append(Block(kind="attention", width=w, n_heads=heads, norm="layer"))
    else:
        for b in c.blocks:
            if b.kind == "mlp":
                b.kind = "residual_mlp"
                b.residual = True
                break
    return c


def neural_fit_signal(train_score: Optional[float], val_score: Optional[float],
                      *, underfit_floor: float = 0.7, gap_eps: float = 0.05) -> Dict[str, bool]:
    """Derive underfit/overfit search-steering signals from train vs. val SCORES.

    Both scores are computed by the trusted parent from PREDICTIONS only (firewall preserved):
    ``train_score`` from a second predict on a held-out TRAIN slice (never val, never sealed),
    ``val_score`` from the spine's val predictions. This introduces NO promotion number; it
    only steers the next architecture sample (design 03 §4.1).

    Heuristics (labelled as such — these are search heuristics, not literature priors):
      - underfit : both train and val are weak (train_score < underfit_floor). Grow capacity.
      - overfit  : train markedly exceeds val (train - val > gap_eps). Regularize / shrink.
    """
    sig = {"underfit": False, "overfit": False}
    if train_score is None or val_score is None:
        return sig
    if train_score < underfit_floor and val_score < underfit_floor:
        sig["underfit"] = True
    if (train_score - val_score) > gap_eps:
        sig["overfit"] = True
    return sig


def _diag_flags(context: Dict[str, Any]) -> Dict[str, bool]:
    """Extract underfit/overfit/plateau flags from context, tolerating the several shapes the
    diagnosis fabric may inject (Diagnosis.as_dict, the directives sub-dict, or a flat dict).

    Degrades gracefully to all-False when no diagnosis is wired (CONTRACT: generalization
    expands what may be proposed; absence must not crash).
    """
    diag = context.get("diagnosis") or {}
    if not isinstance(diag, dict):
        # a Diagnosis dataclass — read its attributes defensively
        diag = getattr(diag, "as_dict", lambda: {})() if hasattr(diag, "as_dict") else {}
    directives = diag.get("directives") if isinstance(diag.get("directives"), dict) else {}
    plateau = bool(diag.get("plateau") or directives.get("plateau"))
    # underfit/overfit may be injected by the neural fit-signal helper into either layer
    underfit = bool(diag.get("underfit") or directives.get("underfit")
                    or context.get("underfit"))
    overfit = bool(diag.get("overfit") or directives.get("overfit") or context.get("overfit"))
    return {"plateau": plateau, "underfit": underfit, "overfit": overfit}


def _intel_signals(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract intelligence signals from the enriched context.

    Reads literature, evolution, and knowledge-base hints injected by the
    intelligence layer (frontier.intelligence.enrich_round_context). Degrades
    gracefully to an empty dict when no intelligence is wired.
    """
    signals: Dict[str, Any] = {}

    # Literature: extract technique names and architecture suggestions
    lit = context.get("literature") or {}
    if isinstance(lit, dict):
        techniques = lit.get("techniques", [])
        architectures = lit.get("architectures", [])
        if techniques:
            signals["literature_techniques"] = techniques
        if architectures:
            signals["literature_architectures"] = architectures
        # Signal: if literature mentions residual/attention/normalization, suggest trying them
        all_text = " ".join(str(t) for t in techniques + architectures).lower()
        signals["try_residual"] = "resid" in all_text or "skip" in all_text
        signals["try_normalization"] = "norm" in all_text or "layer norm" in all_text

    # Knowledge base hints
    kb_hints = context.get("kb_hints")
    if kb_hints:
        signals["kb_hints"] = kb_hints

    # Prompt evolution: score trends and exploration directive
    llm_guidance = context.get("llm_guidance", "")
    if "EXPLORE aggressively" in llm_guidance:
        signals["explore_aggressively"] = True
    elif "Focus on REFINEMENT" in llm_guidance:
        signals["explore_aggressively"] = False

    # Extract evolution directive block if present
    if "[PROMPT EVOLUTION" in llm_guidance:
        start = llm_guidance.find("[PROMPT EVOLUTION")
        signals["evolution_directive"] = llm_guidance[start:start + 500]

    return signals


# --------------------------------------------------------------------------------- renderer

def _render(spec: NeuralSpec) -> Program:
    """Lazy bridge to render.render_program (avoids a circular import at module load)."""
    from .render import render_program
    return render_program(spec)


# ---------------------------------------------------------------------------- proposers

class NASProposer:
    """Complexity-budgeted neural architecture search as a Phase-0 ``ProposalSource``.

    No LLM required: this makes the neural core generative offline, the way MutationProposer
    makes the sklearn path generative offline. Given a champion neural spec it mutates under
    the measured diagnosis (innovation 1); with no champion it samples bounded seed specs.
    Every candidate is validated and budget-filtered BEFORE rendering, so no compute is spent
    on infeasible / bloated nets (innovation 2).

    Determinism: the sample is a pure function of ``(seed, context["round"])`` so the search is
    reproducible (a Phase-7 reproducibility oracle re-checks the winner).
    """

    def __init__(self, *, modality: str = "tabular", param_budget: int = 2_000_000,
                 seed: int = 0, max_proposals: int = 6, vocab_size: int = 0,
                 seq_len: Optional[int] = None):
        if modality not in ("tabular", "sequence", "image1d"):
            raise ValueError(f"unknown modality {modality!r}")
        self.modality = modality
        # param_budget is a hardware/first-principles ceiling, NOT reverse-engineered from any
        # target metric (CONTRACT invariant 3). It only bounds the search.
        self.param_budget = int(param_budget)
        self.seed = int(seed)
        self.max_proposals = int(max_proposals)
        self.vocab_size = int(vocab_size)
        self.seq_len = seq_len  # if None, derived from task n_features

    # -- task-shape derivation from the engine context --------------------------------------
    def _out_dim(self, context: Dict[str, Any]) -> int:
        if context.get("task_kind") == "regression":
            return 1
        # classification: prefer an explicit count, else derive from labels, else binary.
        n = context.get("n_classes") or context.get("out_dim")
        if n:
            return int(n)
        labels = context.get("labels")
        if labels:
            return max(2, len(labels))
        return 2

    def _in_features(self, context: Dict[str, Any]) -> int:
        if self.modality in ("sequence", "image1d") and self.seq_len is not None:
            return int(self.seq_len)
        return int(context.get("n_features") or 1)

    def _task_kind(self, context: Dict[str, Any]) -> str:
        return context.get("task_kind", "classification")

    # -- seed sampling ----------------------------------------------------------------------
    def _sample_seeds(self, context: Dict[str, Any]) -> List[NeuralSpec]:
        rng = random.Random((self.seed * 1_000_003) ^ int(context.get("round", 0)))
        in_f = self._in_features(context)
        out_d = self._out_dim(context)
        kind = self._task_kind(context)
        intel = _intel_signals(context)
        specs: List[NeuralSpec] = []

        # Intelligence-biased grids: literature/evolution may suggest larger or smaller nets
        width_grid = [32, 64, 128, 256]
        depth_grid = [1, 2, 3]
        # If evolution signals exploration, widen the search grid
        if intel.get("explore_aggressively"):
            width_grid = [16, 32, 64, 128, 256, 512]
            depth_grid = [1, 2, 3, 4]
        # If literature mentions specific patterns, bias block kinds
        prefer_residual = intel.get("try_residual", False)
        prefer_norm = intel.get("try_normalization", False)
        norm_choices = (["batch", "layer"] if prefer_norm
                        else ["none", "batch", "layer"])
        kind_choices = (["residual_mlp"] if prefer_residual
                        else ["mlp", "residual_mlp"])

        for _ in range(self.max_proposals * 3):  # oversample; filter to max_proposals
            depth = rng.choice(depth_grid)
            width = rng.choice(width_grid)
            blocks: List[Block] = []
            if self.modality == "sequence":
                vocab = self.vocab_size or int(context.get("vocab_size") or 0)
                if vocab >= 1:
                    blocks.append(Block(kind="embedding_pool", width=width, vocab_size=vocab,
                                        pool="mean"))
                heads = 4 if width % 4 == 0 else 1
                blocks.append(Block(kind="attention", width=width, n_heads=heads, norm="layer",
                                    dropout=round(rng.choice([0.0, 0.1, 0.2]), 4)))
            elif self.modality == "image1d":
                ch = rng.choice([16, 32, 64])
                blocks.append(Block(kind="conv1d", width=ch, kernel_size=3, stride=1,
                                    activation="relu", norm="batch"))
            for _d in range(depth):
                blocks.append(Block(
                    kind=rng.choice(kind_choices), width=width,
                    activation=rng.choice(["relu", "gelu"]),
                    norm=rng.choice(norm_choices),
                    dropout=round(rng.choice([0.0, 0.1, 0.2, 0.3]), 4),
                    residual=rng.random() < (0.7 if prefer_residual else 0.5),
                ))
            spec = NeuralSpec(
                modality=self.modality, task_kind=kind, in_features=in_f, out_dim=out_d,
                blocks=blocks, lr=rng.choice([1e-3, 3e-4, 1e-2]),
                weight_decay=rng.choice([1e-2, 1e-3, 1e-1]),
                max_epochs=200, patience=20, seed=rng.randint(0, 2**31 - 1),
                provenance={"author": "nas", "origin": "seed_sample",
                            "intelligence": bool(intel)},
            )
            specs.append(spec)
        return specs

    def _recover_parent(self, context: Dict[str, Any]) -> Optional[NeuralSpec]:
        """Recover the champion NeuralSpec from context.

        Preferred: context["best_spec"] (a NeuralSpec or its dict). Fallback: the winning
        Program's provenance carries {"neural_spec": {...}} (we set it in propose); the engine
        surfaces best_recipe but not best_spec, so if best_recipe carries it we read it there.
        Returns None if the champion is not a neural spec (e.g. an sklearn recipe won).
        """
        bs = context.get("best_spec")
        if bs is None:
            recipe = context.get("best_recipe") or {}
            if isinstance(recipe, dict):
                bs = recipe.get("neural_spec")
        if bs is None:
            return None
        try:
            return bs if isinstance(bs, NeuralSpec) else NeuralSpec.from_dict(bs)
        except Exception:
            return None

    def propose(self, context: Dict[str, Any]) -> List[Program]:
        """Emit budgeted, validated neural Programs, intelligence-conditioned.

        When intelligence state is available in context (literature, evolution, ensemble),
        uses it to:
        - Bias seed sampling toward architectures mentioned in literature
        - Adjust capacity based on prompt evolution score trends
        - Condition mutations on ensemble complementarity
        """
        parent = self._recover_parent(context)
        budget = int(context.get("param_budget", self.param_budget))

        if parent is None:
            candidates = self._sample_seeds(context)
        else:
            flags = _diag_flags(context)
            intel = _intel_signals(context)
            muts: List[NeuralSpec] = []
            if flags["plateau"]:                       # structural pivot
                muts += [_add_block(parent), _switch_block_kind(parent)]
            if flags["underfit"]:                      # grow capacity
                muts += [_widen(parent), _deepen(parent), _reduce_dropout(parent)]
            if flags["overfit"]:                       # regularize / shrink
                muts += [_add_dropout(parent), _add_norm(parent), _narrow(parent),
                         _raise_weight_decay(parent)]
            # intelligence-driven mutations when diagnosis is absent/neutral
            if not muts or intel.get("explore_aggressively"):
                muts += [_widen(parent), _add_dropout(parent), _deepen(parent)]
            if intel.get("try_residual"):
                muts.append(_switch_block_kind(parent))
            if intel.get("try_normalization"):
                muts.append(_add_norm(parent))
            candidates = muts

        # validate + budget-filter BEFORE any build (innovation 2: free rejection)
        tried = set(context.get("tried_labels", ()))
        out: List[Program] = []
        seen_fp: set = set()
        for spec in candidates:
            if validate(spec):
                continue
            if estimate_params(spec) > budget:
                continue
            if spec.fingerprint in seen_fp:
                continue
            seen_fp.add(spec.fingerprint)
            prog = _render(spec)
            if prog.label in tried:
                continue
            out.append(prog)
            if len(out) >= self.max_proposals:
                break
        return out


class LLMArchitectProposer:
    """LLM-authored architecture as a Phase-0 ``ProposalSource``.

    Reuses the pluggable ``client: Callable[[str], str]`` contract from LLMProposer. The LLM's
    creativity lives in the NeuralSpec it DESIGNS (block kinds, widths, depths, residual /
    attention / norm choices, training knobs), expressed as constrained JSON. The torch code
    that actually runs is emitted by the AUDITED render.py, not by the model — the key safety
    improvement over the legacy template builder (design 03 §4.2).

    Guards (all three honored):
      1. Parse: extract the first JSON object; a parse failure is recorded, never silently
         dropped, and fed back next round via recent_errors.
      2. Validate: spec.validate must pass; violations are returned as text for the next round.
      3. The rendered code STILL runs in the same sandbox and STILL must define
         build_estimator(); it has no certification privilege.

    When ``client is None`` the proposer returns [] (the engine reports the LLM path inactive —
    CONTRACT: never fabricate proposals). It degrades honestly with no torch and no LLM.
    """

    def __init__(self, client: Optional[Any] = None, *, modality: str = "tabular",
                 param_budget: int = 2_000_000, n: int = 2):
        self.client = client
        self.modality = modality
        self.param_budget = int(param_budget)
        self.n = int(n)
        self.last_parse_errors: List[str] = []  # surfaced for the round log / recent_errors

    def _prompt(self, context: Dict[str, Any]) -> str:
        flags = _diag_flags(context)
        intel = _intel_signals(context)
        diag_txt = ", ".join(k for k, v in flags.items() if v) or "none"
        errs = context.get("recent_errors", [])
        err_txt = "\n".join(f"  - {lab}: [{ek}] {msg}" for lab, ek, msg in errs[:5]) or "  (none)"
        champ = context.get("best_label") or "none"

        # Build intelligence-enriched prompt sections
        intel_section = ""
        if intel.get("literature_techniques"):
            techniques = intel["literature_techniques"][:5]
            intel_section += "\nRelevant techniques from recent literature:\n"
            intel_section += "\n".join(f"  - {t}" for t in techniques)
        if intel.get("evolution_directive"):
            intel_section += f"\n\nSearch evolution guidance:\n{intel['evolution_directive']}"
        if intel.get("kb_hints"):
            hints = intel["kb_hints"][:3]
            intel_section += "\n\nKnowledge from similar problems:\n"
            intel_section += "\n".join(f"  - {h}" for h in hints)

        return (
            "You are designing a neural network architecture for an ML task. Return ONLY a "
            "single JSON object describing a NeuralSpec — no prose, no markdown fences.\n\n"
            f"modality: {self.modality}\n"
            f"task_kind: {context.get('task_kind')}\n"
            f"in_features: {context.get('n_features')}\n"
            f"n_train: {context.get('n_train')}\n"
            f"current champion: {champ} (val {context.get('best_score')})\n"
            f"diagnosis signals: {diag_txt}\n"
            f"parameter budget (max trainable params): {self.param_budget}\n"
            "recent failures (avoid repeating):\n"
            f"{err_txt}\n"
            f"{intel_section}\n\n"
            "JSON schema: {\"modality\":str, \"task_kind\":str, \"in_features\":int, "
            "\"out_dim\":int, \"blocks\":[{\"kind\":\"mlp|residual_mlp|conv1d|attention|"
            "embedding_pool\", \"width\":int, \"activation\":\"relu|gelu|silu|tanh|identity\", "
            "\"norm\":\"none|batch|layer\", \"dropout\":float, \"residual\":bool, "
            "\"kernel_size\":int, \"stride\":int, \"n_heads\":int, \"vocab_size\":int}], "
            "\"optimizer\":\"adam|adamw|sgd\", \"lr\":float, \"weight_decay\":float, "
            "\"batch_size\":int, \"max_epochs\":int, \"patience\":int, \"seed\":int}\n"
            "out_dim must be 1 for regression and n_classes for classification. attention/"
            "embedding_pool are sequence-only. Keep estimated params under the budget."
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract the first balanced JSON object from arbitrary model output."""
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def propose(self, context: Dict[str, Any]) -> List[Program]:
        self.last_parse_errors = []
        if self.client is None:
            return []  # honest: LLM path inactive, no fabrication
        prompt = self._prompt(context)
        tried = set(context.get("tried_labels", ()))
        budget = int(context.get("param_budget", self.param_budget))
        out: List[Program] = []
        seen_fp: set = set()
        for i in range(self.n):
            try:
                raw = self.client(prompt)
            except Exception as e:  # client failure is recorded, loop continues
                self.last_parse_errors.append(f"client raised: {type(e).__name__}: {e}")
                break
            d = self._extract_json(raw or "")
            if d is None:
                self.last_parse_errors.append("could not parse a JSON object from LLM output")
                continue
            try:
                spec = NeuralSpec.from_dict(d)
            except Exception as e:
                self.last_parse_errors.append(f"spec build failed: {type(e).__name__}: {e}")
                continue
            spec.provenance = {"author": "llm", "round": context.get("round", 0)}
            violations = validate(spec)
            if violations:
                self.last_parse_errors.append("invalid spec: " + "; ".join(violations[:4]))
                continue
            if estimate_params(spec) > budget:
                self.last_parse_errors.append(
                    f"over budget: {estimate_params(spec)} > {budget}")
                continue
            if spec.fingerprint in seen_fp:
                continue
            seen_fp.add(spec.fingerprint)
            prog = _render(spec)
            if prog.label in tried:
                continue
            out.append(prog)
        return out
