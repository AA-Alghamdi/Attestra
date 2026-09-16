"""Phase 8: compounding cross-task knowledge (the flywheel).

The autoresearcher should get *cheaper and better the more tasks it sees*. Phase 0 starts
every task cold: the same seeds, the same mutation order, no memory of what worked on the
last twenty datasets. This module adds the durable memory and the two mechanisms that turn it
into compounding advantage:

  1. KnowledgeBase  -- a durable research KB keyed by (problem_type, dataset_fingerprint),
     storing one atomic record per certified/evaluated candidate:
        (recipe descriptor, realized val gain over the round's incumbent, cost in wall-seconds).
     Persisted as append-only JSONL (the CONTRACT's "results files are append-only" rule),
     one self-describing JSON object per line, so concurrent runs never corrupt each other and
     a crash mid-write loses at most the last line.

  2. RetrievalProposer -- WARM-STARTS a new task by retrieving the best historical recipes for
     *similar* fingerprints (cosine similarity over a fixed, interpretable fingerprint vector)
     and seeding them as Programs with source="retrieval". This is transfer: run N on a dataset
     similar to ones already seen skips the cold-start search and proposes proven recipes first.

  3. LinUCBRanker + KnowledgeProposer -- a REAL LinUCB contextual bandit (Li et al. 2010,
     "A Contextual-Bandit Approach to Personalized News Article Recommendation") over the
     feature vector phi = [task_descriptor (x) recipe_descriptor]. It RANKS the round's
     proposals by upper-confidence-bound reward and UPDATES its (A, b) sufficient statistics
     from realized outcomes. This wires the LinUCB that the roadmap notes is currently only
     *printed* (orchestrator.py:251-259) into the actual proposal ordering, so ranking reorders
     what the engine tries first.

INTEGRITY (CONTRACT invariants preserved):
  - The KB / bandit NEVER promote. They only change what is PROPOSED and in what ORDER
    (invariant 4: generalization expands/orders proposals, never what may promote). The frozen
    certifier in certify.py is still the only thing that touches the sealed test and decides.
  - Records store the *realized val gain* (a selection-set number the trusted parent already
    computed via certify.score_val), never a sealed/relabeled number. The reward signal the
    bandit learns from is the same honest val score the engine selects on.
  - Retrieval and bandit are SEEDS/ORDERING (invariant 3): the deterministic fallback when the
    KB is empty is "no warm start, identity order", which degrades to exactly Phase-0 behavior.
  - No new JIT-sensitive globals; no torch; numpy + stdlib only.


# === WIRING ===
# The integrator plugs this into ResearchEngine WITHOUT editing engine.py, by:
#
# (A) Construct shared state once, before the run (so it persists across tasks):
#         from frontier.knowledge import KnowledgeBase, RetrievalProposer, LinUCBRanker, \
#                                        KnowledgeProposer, task_fingerprint, recipe_descriptor
#         kb     = KnowledgeBase("frontier_kb.jsonl")        # durable, append-only
#         ranker = LinUCBRanker(alpha=1.0)                   # one bandit across all tasks
#         ranker.load_from_kb(kb)                            # replay history -> warm A,b
#
# (B) Add the RetrievalProposer to the proposer list, FIRST, so warm-start seeds lead:
#         proposers = [RetrievalProposer(kb), SeedProposer(), MutationProposer(),
#                      LLMProposer(cfg.llm_client)]
#         engine = ResearchEngine(cfg, proposers=proposers)
#     RetrievalProposer.propose(context) reads context["task_fingerprint"] (see (D)); when the
#     KB has similar fingerprints it returns retrieval Programs, else [] (cold -> Phase-0 floor).
#
# (C) The engine's per-round dedup loop already collects `proposals: list[Program]`. The
#     integrator reorders that list with the ranker right before the sandbox loop:
#         proposals = ranker.rank(proposals, context)        # UCB order; ties keep input order
#     This is a stable reorder of the SAME proposals; it changes try-order, not membership.
#
# (D) The engine builds `context` each round (engine.py:109-120). The integrator adds two keys
#     so retrieval/ranking have the task descriptor (Program already carries provenance["recipe"]):
#         context["task_descriptor"]  = task_descriptor(task)          # dict
#         context["task_fingerprint"] = task_fingerprint(task)         # np.ndarray (fixed dim)
#     task_descriptor/task_fingerprint depend ONLY on the Task object the engine already holds.
#
# (E) After each candidate is scored on VAL (engine.py:145-152), the integrator records the
#     outcome and updates the bandit from the SAME honest val number:
#         gain = (score - incumbent_before) if incumbent_before is not None else score
#         kb.record(problem_type=task.kind, fingerprint=context["task_fingerprint"],
#                   task_descriptor=context["task_descriptor"],
#                   recipe=p.provenance.get("recipe", {"label": p.label, "source": p.source}),
#                   val_gain=gain, val_score=score, cost_seconds=res.wall_seconds,
#                   program_id=p.id)
#         ranker.update(context, p, reward=score)            # bounded reward (see _clip_reward)
#     On failure (res.ok False) the integrator records val_gain=0, cost_seconds=res.wall_seconds
#     and ranker.update(..., reward=0.0) so the bandit learns to deprioritize failing recipes.
#
# Shapes: task_fingerprint -> (FINGERPRINT_DIM,) float; recipe_descriptor -> (RECIPE_DIM,) float;
# phi (task (x) recipe context for LinUCB) -> (CONTEXT_DIM,) float, CONTEXT_DIM fixed & shared.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .program import Program
from .proposers import _BASES, make_code, recipe_label


# --------------------------------------------------------------------------- descriptors
# Fixed, interpretable, order-stable vocabularies. Keeping these frozen is what makes the
# LinUCB context vector a stable dimension across tasks and runs -- a bandit cannot learn if
# its feature layout shifts between updates. New bases append at the end (never reorder).

_BASE_VOCAB: Tuple[str, ...] = (
    "hist_gbm", "rf", "logreg", "svc_rbf", "ridge", "gbr",
)
# enhancer flags the recipe space exposes (proposers.make_code understands these)
_ENHANCERS: Tuple[str, ...] = ("scale", "poly", "target_log")

RECIPE_DIM: int = len(_BASE_VOCAB) + len(_ENHANCERS) + 1  # +1 bias term
# task descriptor: [kind_is_clf, log10(n), log10(d), class_imbalance_or_target_cv, n_classes_norm]
TASK_DIM: int = 5
FINGERPRINT_DIM: int = TASK_DIM
# LinUCB context: task (x) "salient recipe coords" would explode dimension; instead we use the
# disjoint-LinUCB layout phi = [task_desc , recipe_desc , task_desc*scalar_recipe_summary].
# Concretely: concatenation of task and recipe descriptors plus their interaction summary.
CONTEXT_DIM: int = TASK_DIM + RECIPE_DIM + 1


def task_descriptor(task) -> dict:
    """Human-readable task descriptor (also the source for the numeric fingerprint).

    Depends only on the Task object the engine already holds. Captures the coarse problem
    shape that makes two datasets "similar" for transfer: kind, size, dimensionality, and a
    difficulty proxy (class imbalance for clf; target coefficient-of-variation for reg).
    """
    X = np.asarray(task.X, dtype=float)
    n, d = (X.shape[0], X.shape[1]) if X.ndim == 2 else (len(X), 1)
    if task.kind == "classification":
        _, counts = np.unique(np.asarray(task.y), return_counts=True)
        n_classes = int(len(counts))
        # imbalance in [0,1): 0 = perfectly balanced; ->1 as one class dominates.
        frac = counts / counts.sum()
        imbalance = float(1.0 - n_classes * frac.min()) if n_classes > 1 else 0.0
        imbalance = max(0.0, min(1.0, imbalance))
        difficulty = imbalance
    else:
        y = np.asarray(task.y, dtype=float)
        mu = float(np.mean(np.abs(y))) + 1e-12
        difficulty = float(np.std(y) / mu)        # coefficient of variation (scale-free)
        n_classes = 0
    return {
        "kind": task.kind,
        "n": int(n),
        "d": int(d),
        "difficulty": float(difficulty),
        "n_classes": int(n_classes),
    }


def task_fingerprint(task) -> np.ndarray:
    """Fixed-dim numeric fingerprint for similarity-based retrieval and bandit context.

    Returns shape (FINGERPRINT_DIM,). Coordinates are bounded/log-scaled so cosine and dot
    products are meaningful across datasets of very different sizes.
    """
    desc = task_descriptor(task)
    return _descriptor_to_vec(desc)


def _descriptor_to_vec(desc: dict) -> np.ndarray:
    kind_is_clf = 1.0 if desc.get("kind") == "classification" else 0.0
    n = max(1, int(desc.get("n", 1)))
    d = max(1, int(desc.get("d", 1)))
    difficulty = float(desc.get("difficulty", 0.0))
    n_classes = int(desc.get("n_classes", 0))
    return np.array([
        kind_is_clf,
        math.log10(n + 1.0) / 6.0,          # ~[0,1] for n up to 1e6
        math.log10(d + 1.0) / 4.0,          # ~[0,1] for d up to 1e4
        math.tanh(difficulty),              # squashed difficulty proxy
        min(n_classes, 20) / 20.0,          # normalized class count (0 for regression)
    ], dtype=float)


def recipe_descriptor(recipe: dict) -> np.ndarray:
    """One-hot base + enhancer flags + bias. Shape (RECIPE_DIM,).

    A recipe is the proposers.py dict {base, scale, poly, target_log}. Programs that are not
    recipe-shaped (raw LLM code) get an all-zero base block plus the bias term, so they still
    receive a valid (low-information) context rather than crashing the bandit.
    """
    v = np.zeros(RECIPE_DIM, dtype=float)
    base = (recipe or {}).get("base")
    if base in _BASE_VOCAB:
        v[_BASE_VOCAB.index(base)] = 1.0
    off = len(_BASE_VOCAB)
    for j, flag in enumerate(_ENHANCERS):
        val = (recipe or {}).get(flag)
        # poly is an int degree; scale/target_log are bool. Normalize to a [0,1]-ish scalar.
        if flag == "poly":
            v[off + j] = min(float(val or 0), 3.0) / 3.0
        else:
            v[off + j] = 1.0 if val else 0.0
    v[-1] = 1.0  # bias
    return v


def _recipe_of(program: Program) -> dict:
    """Best-effort recipe dict for a Program (recipe-shaped or raw)."""
    rec = program.provenance.get("recipe") if isinstance(program.provenance, dict) else None
    if isinstance(rec, dict) and rec:
        return rec
    return {"label": program.label, "source": program.source}


def context_vector(task_desc: dict, recipe: dict) -> np.ndarray:
    """LinUCB context phi(task, recipe). Shape (CONTEXT_DIM,).

    Layout: [ task_descriptor | recipe_descriptor | <task.difficulty * recipe.enhancer_mass> ].
    The last coordinate is a deliberate interaction term: feature-engineering enhancers tend to
    pay off more on harder/higher-CV targets, and a linear bandit needs the cross-term made
    explicit (it cannot form products of its own inputs).
    """
    t = _descriptor_to_vec(task_desc)
    r = recipe_descriptor(recipe)
    enhancer_mass = float(np.sum(r[len(_BASE_VOCAB):len(_BASE_VOCAB) + len(_ENHANCERS)]))
    interaction = float(math.tanh(task_desc.get("difficulty", 0.0))) * enhancer_mass
    return np.concatenate([t, r, np.array([interaction], dtype=float)])


# --------------------------------------------------------------------------- KnowledgeBase

@dataclass
class KBRecord:
    """One atomic KB entry: a recipe tried on a fingerprinted task and what it earned."""
    problem_type: str
    fingerprint: List[float]
    task_descriptor: dict
    recipe: dict
    recipe_label: str
    val_gain: float          # realized gain over the round incumbent (honest, val-set)
    val_score: float         # the absolute val score (for ranking retrieval seeds)
    cost_seconds: float
    program_id: str
    ts: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "problem_type": self.problem_type,
            "fingerprint": [float(x) for x in self.fingerprint],
            "task_descriptor": self.task_descriptor,
            "recipe": self.recipe,
            "recipe_label": self.recipe_label,
            "val_gain": float(self.val_gain),
            "val_score": float(self.val_score),
            "cost_seconds": float(self.cost_seconds),
            "program_id": self.program_id,
            "ts": float(self.ts),
        }, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "KBRecord":
        return KBRecord(
            problem_type=d["problem_type"],
            fingerprint=list(d.get("fingerprint", [])),
            task_descriptor=d.get("task_descriptor", {}),
            recipe=d.get("recipe", {}),
            recipe_label=d.get("recipe_label", ""),
            val_gain=float(d.get("val_gain", 0.0)),
            val_score=float(d.get("val_score", 0.0)),
            cost_seconds=float(d.get("cost_seconds", 0.0)),
            program_id=d.get("program_id", ""),
            ts=float(d.get("ts", 0.0)),
        )


class KnowledgeBase:
    """Durable, append-only research KB keyed by (problem_type, dataset_fingerprint).

    Backed by a JSONL file (one JSON object per line). Append-only per the CONTRACT: we never
    rewrite existing lines, so a concurrent reader/writer or a crash mid-write loses at most the
    final partial line (which load() skips). An in-memory mirror is kept for fast retrieval.

    If path is None the KB is purely in-memory (handy for tests / ephemeral runs).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._records: List[KBRecord] = []
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            self._load(path)

    # -- persistence -------------------------------------------------------
    def _load(self, path: str) -> None:
        recs: List[KBRecord] = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(KBRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    # tolerate a torn last line from a crashed writer; never crash on load.
                    continue
        self._records = recs

    def record(self, *, problem_type: str, fingerprint, task_descriptor: dict,
               recipe: dict, val_gain: float, val_score: float, cost_seconds: float,
               program_id: str = "") -> KBRecord:
        """Append one atomic outcome. Returns the stored KBRecord.

        `val_gain`/`val_score` MUST be the honest validation-set numbers the trusted parent
        computed (never a sealed or relabeled number) -- this is the bandit's reward source.
        """
        rec = KBRecord(
            problem_type=str(problem_type),
            fingerprint=[float(x) for x in np.asarray(fingerprint, dtype=float).ravel()],
            task_descriptor=dict(task_descriptor),
            recipe=dict(recipe),
            recipe_label=recipe.get("label") or recipe_label(recipe) if _is_recipe(recipe)
                         else str(recipe.get("label", "")),
            val_gain=float(val_gain),
            val_score=float(val_score),
            cost_seconds=float(cost_seconds),
            program_id=str(program_id),
            ts=time.time(),
        )
        with self._lock:
            self._records.append(rec)
            if self.path:
                # atomic single-line append: open in append mode, write one line, flush+fsync.
                with open(self.path, "a") as fh:
                    fh.write(rec.to_json() + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        return rec

    def all_records(self) -> List[KBRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, problem_type: str, fingerprint, *, k: int = 4,
                 min_similarity: float = 0.90, min_gain: float = 1e-9) -> List[KBRecord]:
        """Return up to k historical recipes that worked on SIMILAR fingerprints.

        Similarity is cosine over the fixed fingerprint vector, restricted to the same
        problem_type (a classification recipe is meaningless for regression and vice versa).
        We keep only records with a positive realized gain (they actually helped), then rank
        by a similarity-weighted gain, and DEDUP by recipe_label keeping the best instance.
        Returns [] when nothing clears the bar -> caller falls back to cold Phase-0 seeds.
        """
        q = np.asarray(fingerprint, dtype=float).ravel()
        qn = np.linalg.norm(q)
        if qn == 0.0:
            return []
        scored: List[Tuple[float, KBRecord]] = []
        with self._lock:
            recs = list(self._records)
        for rec in recs:
            if rec.problem_type != problem_type:
                continue
            if rec.val_gain < min_gain:
                continue
            v = np.asarray(rec.fingerprint, dtype=float).ravel()
            if v.shape != q.shape:
                continue
            vn = np.linalg.norm(v)
            if vn == 0.0:
                continue
            sim = float(np.dot(q, v) / (qn * vn))
            if sim < min_similarity:
                continue
            scored.append((sim * rec.val_gain, rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: List[KBRecord] = []
        seen: set = set()
        for _, rec in scored:
            if rec.recipe_label in seen:
                continue
            seen.add(rec.recipe_label)
            out.append(rec)
            if len(out) >= k:
                break
        return out


def _is_recipe(d: dict) -> bool:
    return isinstance(d, dict) and "base" in d


# --------------------------------------------------------------------------- RetrievalProposer

class RetrievalProposer:
    """Warm-start a new task with the best historical recipes for similar fingerprints.

    A ProposalSource (satisfies the proposers.ProposalSource Protocol). On a repeat or similar
    fingerprint it emits recipe-shaped Programs with source="retrieval", ordered best-first, so
    the engine tries proven recipes before cold seeds. On a cold/empty KB it returns [] and the
    rest of the proposer list (Seed/Mutation/LLM) supplies the Phase-0 floor unchanged.
    """

    def __init__(self, kb: KnowledgeBase, *, k: int = 4, min_similarity: float = 0.90):
        self.kb = kb
        self.k = k
        self.min_similarity = min_similarity

    def propose(self, context: dict) -> List[Program]:
        kind = context.get("task_kind")
        fp = context.get("task_fingerprint")
        if kind is None or fp is None:
            return []
        tried = set(context.get("tried_labels", ()))
        recs = self.kb.retrieve(kind, fp, k=self.k, min_similarity=self.min_similarity)
        out: List[Program] = []
        for rec in recs:
            if not _is_recipe(rec.recipe):
                continue                          # only recipe-shaped entries are re-renderable
            try:
                code = make_code(rec.recipe, kind)
            except KeyError:
                continue                          # base no longer in the catalog -> skip
            label = recipe_label(rec.recipe)
            if label in tried:
                continue
            out.append(Program(
                code=code, source="retrieval", label=label,
                provenance={"recipe": dict(rec.recipe),
                            "retrieved_from": rec.program_id,
                            "prior_val_gain": rec.val_gain,
                            "prior_val_score": rec.val_score},
            ))
        return out


# --------------------------------------------------------------------------- LinUCB

@dataclass
class LinUCBRanker:
    """Disjoint-free LinUCB contextual bandit over phi(task, recipe). Shape (CONTEXT_DIM,).

    Maintains the ridge-regression sufficient statistics A = lambda*I + sum phi phi^T and
    b = sum reward*phi (Li et al. 2010). The UCB score for a proposal is
        theta_hat^T phi + alpha * sqrt(phi^T A^{-1} phi),
    i.e. exploit (predicted reward) + explore (predictive std). `rank` orders proposals by this
    score (stable: ties keep input order, so an empty bandit reproduces Phase-0 order exactly).
    `update` folds a realized reward into (A, b).

    Reward is the honest val score in [0,1]-ish; we clip to [0,1] so a stray regression r2 below
    0 or an outlier cannot blow up the linear fit. This NEVER promotes; it only orders proposals.
    """

    alpha: float = 1.0
    ridge: float = 1.0
    dim: int = CONTEXT_DIM
    A: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    n_updates: int = 0

    def __post_init__(self):
        if self.A is None:
            self.A = self.ridge * np.eye(self.dim)
        if self.b is None:
            self.b = np.zeros(self.dim)

    # -- core math ---------------------------------------------------------
    def _theta(self) -> np.ndarray:
        # solve A theta = b  (more stable than explicit inverse)
        return np.linalg.solve(self.A, self.b)

    def _ucb(self, phi: np.ndarray, A_inv: np.ndarray, theta: np.ndarray) -> float:
        mean = float(theta @ phi)
        var = float(phi @ A_inv @ phi)
        var = max(var, 0.0)
        return mean + self.alpha * math.sqrt(var)

    @staticmethod
    def _clip_reward(r: float) -> float:
        if not math.isfinite(r):
            return 0.0
        return float(min(1.0, max(0.0, r)))

    # -- public API --------------------------------------------------------
    def score(self, context: dict, program: Program) -> float:
        """UCB score for a single (task, recipe) pair. Higher = try sooner."""
        td = context.get("task_descriptor") or _td_from_context(context)
        phi = context_vector(td, _recipe_of(program))
        A_inv = np.linalg.inv(self.A)
        return self._ucb(phi, A_inv, self._theta())

    def rank(self, proposals: Sequence[Program], context: dict) -> List[Program]:
        """Stably reorder proposals by descending UCB. Ties preserve input order.

        Empty/untrained bandit: theta_hat = 0 and the exploration bonus is monotone in
        phi^T A^{-1} phi, so order is well-defined and deterministic; with the symmetric seed
        recipes it equals the input order, matching Phase-0 behavior until the KB learns.
        """
        if not proposals:
            return list(proposals)
        td = context.get("task_descriptor") or _td_from_context(context)
        A_inv = np.linalg.inv(self.A)
        theta = self._theta()
        scored = []
        for i, p in enumerate(proposals):
            phi = context_vector(td, _recipe_of(p))
            scored.append((-self._ucb(phi, A_inv, theta), i, p))
        scored.sort(key=lambda t: (t[0], t[1]))   # primary: -ucb; secondary: original index
        return [p for _, _, p in scored]

    def update(self, context: dict, program: Program, reward: float) -> None:
        """Fold one realized outcome into (A, b). reward is the honest val score."""
        td = context.get("task_descriptor") or _td_from_context(context)
        phi = context_vector(td, _recipe_of(program))
        r = self._clip_reward(reward)
        self.A += np.outer(phi, phi)
        self.b += r * phi
        self.n_updates += 1

    def load_from_kb(self, kb: KnowledgeBase) -> int:
        """Replay the KB to warm-start (A, b). Returns the number of records folded in.

        Uses each record's absolute val_score as the reward and reconstructs the context from
        the stored task_descriptor + recipe. Lets a fresh process inherit the whole history's
        bandit state, so ranking compounds across runs, not just within one run.
        """
        n = 0
        for rec in kb.all_records():
            td = rec.task_descriptor or {}
            phi = context_vector(td, rec.recipe)
            r = self._clip_reward(rec.val_score)
            self.A += np.outer(phi, phi)
            self.b += r * phi
            n += 1
        self.n_updates += n
        return n


def _td_from_context(context: dict) -> dict:
    """Recover a task descriptor from a context that only carries kind/n_features/fingerprint.

    The wiring block asks the integrator to put context['task_descriptor'] in; this is the
    graceful fallback if only the fingerprint or the coarse fields are present, so ranking still
    works rather than throwing.
    """
    fp = context.get("task_fingerprint")
    if fp is not None:
        fp = np.asarray(fp, dtype=float).ravel()
        if fp.shape[0] == FINGERPRINT_DIM:
            # invert the (lossy) fingerprint encoding well enough for the context vector.
            kind = "classification" if fp[0] >= 0.5 else "regression"
            n = int(round(10 ** (fp[1] * 6.0)))
            d = int(round(10 ** (fp[2] * 4.0)))
            difficulty = float(np.arctanh(min(0.999, max(-0.999, fp[3]))))
            n_classes = int(round(fp[4] * 20))
            return {"kind": kind, "n": n, "d": d, "difficulty": difficulty,
                    "n_classes": n_classes}
    return {"kind": context.get("task_kind", "classification"),
            "n": int(context.get("n_train", 1)),
            "d": int(context.get("n_features", 1)),
            "difficulty": 0.0, "n_classes": 0}


# --------------------------------------------------------------------------- KnowledgeProposer

class KnowledgeProposer:
    """The single object the integrator drops in: warm-start retrieval + bandit ranking.

    Composes a RetrievalProposer (transfer seeds) with a LinUCBRanker (ordering) and shares the
    KnowledgeBase. It exposes the ProposalSource interface (`propose`) AND a `rank`/`record`
    surface so an integrator can use it as a one-stop knowledge layer:

      - propose(context)            -> retrieval warm-start Programs (or [] when cold)
      - rank(proposals, context)    -> bandit-ordered proposals (stable; identity when cold)
      - record_outcome(...)         -> append to KB + update the bandit in one call

    This keeps the engine edit surface to: add this to the proposer list, call .rank on the
    deduped proposal list, and call .record_outcome after each val score.
    """

    def __init__(self, kb: Optional[KnowledgeBase] = None, *, alpha: float = 1.0,
                 k: int = 4, min_similarity: float = 0.90, warm_start_bandit: bool = True):
        self.kb = kb if kb is not None else KnowledgeBase(None)
        self.retrieval = RetrievalProposer(self.kb, k=k, min_similarity=min_similarity)
        self.ranker = LinUCBRanker(alpha=alpha)
        if warm_start_bandit:
            self.ranker.load_from_kb(self.kb)

    def propose(self, context: dict) -> List[Program]:
        return self.retrieval.propose(context)

    def rank(self, proposals: Sequence[Program], context: dict) -> List[Program]:
        return self.ranker.rank(proposals, context)

    def record_outcome(self, context: dict, program: Program, *, val_score: float,
                       incumbent_before: Optional[float], cost_seconds: float,
                       ok: bool = True) -> None:
        """Persist the outcome and update the bandit from the SAME honest val number.

        On a failed candidate (ok False) we record zero gain/score so the bandit and KB learn to
        deprioritize recipes that crash, without inventing a fake score.
        """
        score = float(val_score) if ok else 0.0
        gain = (score - incumbent_before) if (ok and incumbent_before is not None) else (
            score if ok else 0.0)
        td = context.get("task_descriptor") or _td_from_context(context)
        fp = context.get("task_fingerprint")
        if fp is None:
            fp = _descriptor_to_vec(td)
        self.kb.record(
            problem_type=context.get("task_kind", td.get("kind", "classification")),
            fingerprint=fp,
            task_descriptor=td,
            recipe=_recipe_of(program),
            val_gain=gain,
            val_score=score,
            cost_seconds=float(cost_seconds),
            program_id=program.id,
        )
        self.ranker.update(context, program, reward=score)
