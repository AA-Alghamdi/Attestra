"""Phase 4 -- the agentic coding loop (executor, not one-shot proposer).

A one-shot proposer emits code and walks away: if the first run fails, the candidate is
lost. A real research engineer instead writes -> runs -> reads the traceback -> fixes ->
re-runs until the pipeline works. This module turns proposal into that loop.

The substrate is the Phase-0 sandbox (`frontier.sandbox.run_program`): every attempt --
the original and every repair -- is executed out-of-process, returns predictions only, and
is scored/certified later by the trusted parent. The repair loop NEVER computes a metric and
NEVER touches the sealed test; it only gets a candidate to the point where it RUNS. The
existing selection + sealed-certification path in `engine.py` is unchanged.

Two driving capabilities:

  * `repair_loop(program, ...)`: given a Program that fails in the sandbox (any RunResult
    with ok=False and a typed `error_kind` + `error`), ask the repairer to FIX the code,
    re-run, and iterate up to `k` times. The full error history is threaded into each fix
    attempt so the repairer does not loop on the same mistake.

  * `AgenticProposer`: a `ProposalSource` (drop-in for SeedProposer/MutationProposer/
    LLMProposer). It authors candidate code from scratch using the round's diagnosis
    context, then runs each candidate through `repair_loop` so what it returns to the engine
    is code that already executes. It can also wrap any base proposer to make IT agentic.

Powered by LLMs, not competing
------------------------------
The repairer is pluggable: `llm_client: Callable[[str], str] | None`.
  * With a client, the LLM reads the error + history and rewrites the code (open-ended, the
    strong path; improves for free as base models improve).
  * With None, a DETERMINISTIC repair heuristic runs: it parses the typed error and applies
    conservative, well-understood source transforms (fix a wrong import path, define a
    missing name, strip an unsupported estimator kwarg, synthesize a missing
    build_estimator). These are SEEDS/FALLBACKS -- they are never a promotion-bearing or
    search-bounding decision (the frozen certifier still promotes; the sandbox still runs the
    fixed code). They exist so the loop demonstrably repairs a broken Program with no model
    wired in, and so an LLM outage degrades to "fewer fixes," not "silent fabrication."

# === WIRING ===
# The integrator plugs this into the existing spine in one of two ways; both are additive and
# require NO change to engine.py / proposers.py / certify.py / sandbox.py.
#
# (A) As an extra proposal source (recommended). Pass it in the proposers list:
#
#       from frontier.agentic import AgenticProposer
#       from frontier.engine import ResearchEngine, EngineConfig
#       from frontier.proposers import SeedProposer, MutationProposer, LLMProposer
#       cfg = EngineConfig(rounds=3, llm_client=my_client)   # my_client: prompt->code, or None
#       engine = ResearchEngine(cfg, proposers=[
#           SeedProposer(),
#           MutationProposer(),
#           AgenticProposer(llm_client=cfg.llm_client),       # authors + self-repairs
#       ])
#       result = engine.run(task)
#
#     Contract honored: AgenticProposer.propose(context) -> list[Program], where `context`
#     is the EXACT dict engine.py builds each round (keys: task_kind, n_features, n_train,
#     round, tried_labels:set, best_label, best_score, best_id, best_recipe, recent_errors:
#     list[(label, error_kind, msg)]). It returns only Programs whose code ALREADY ran in the
#     sandbox during repair (or, if all repairs failed, the best-effort author attempt so the
#     engine still records the typed failure -- never a fabricated success). The engine then
#     runs each returned Program on VAL itself (its normal path); a repaired Program simply
#     re-runs cleanly there. The repaired code carries provenance {"repair": {...}} so the
#     diagnosis trail can see the loop fired.
#
# (B) As a wrapper to make ANY proposer agentic:
#
#       AgenticProposer(base=MutationProposer(), llm_client=client)
#
#     It calls base.propose(context), then repair_loop()s each emitted Program. Use this to
#     give the offline mutation/seed proposers self-repair without writing-from-scratch.
#
# The repair loop needs to execute candidates, so it must be given the training arrays + eval
# split + kind. AgenticProposer pulls those from the context via the optional keys
# "X_train", "y_train", "X_probe", "kind" if the integrator chooses to add them (see
# `attach_probe_to_context`); if they are absent it falls back to a tiny self-contained
# synthetic probe matched to task_kind/n_features so repair can still verify "does it run".
# This keeps AgenticProposer a pure ProposalSource (no engine.py edit required): the engine's
# own VAL run remains the authoritative execution; the probe only drives the repair iteration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import sandbox
from .program import Program, RunResult


# --------------------------------------------------------------------------- repair record

@dataclass
class RepairStep:
    """One iteration of the repair loop: what was tried and what came back."""
    attempt: int                 # 0 = original code, 1.. = repairs
    source: str                  # "original" | "llm" | "deterministic"
    error_kind: str              # typed error from the sandbox for THIS attempt ("" if ok)
    error: str                   # error message for this attempt ("" if ok)
    ok: bool                     # did this attempt run in the sandbox?
    note: str = ""               # what the repairer changed (heuristic name / "llm rewrite")


@dataclass
class RepairResult:
    """Outcome of repair_loop: the (possibly fixed) program + the run + the history."""
    program: Program             # final program (repaired if a fix succeeded, else last attempt)
    run: RunResult               # sandbox RunResult for `program` (ok=True iff repaired/ran)
    repaired: bool               # True iff a fix turned a failing program into a running one
    steps: List[RepairStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.run.ok

    def history_tuples(self) -> List[Tuple[int, str, str]]:
        """(attempt, error_kind, error) for every failed attempt -- threaded into prompts."""
        return [(s.attempt, s.error_kind, s.error) for s in self.steps if not s.ok]


# --------------------------------------------------------------------------- the repairer

class CodeRepairer:
    """Turns a failing candidate's code + typed error into a candidate FIX (new code).

    Two backends behind one interface (`fix`):
      - LLM: `llm_client(prompt) -> code`. The prompt carries the current code, the typed
        error, and the full failed-attempt history so the model does not repeat itself.
      - Deterministic fallback (client is None OR the client raised/returned junk): a small
        set of conservative, named source transforms keyed off the error taxonomy.

    Returns (new_code, source, note) or None if it cannot produce a *different* candidate.
    `source` is "llm" or "deterministic"; `note` describes the change for the trail.
    """

    def __init__(self, llm_client: Optional[Callable[[str], str]] = None):
        self.llm_client = llm_client

    # -- public ------------------------------------------------------------------------
    def fix(self, code: str, error_kind: str, error: str,
            history: List[Tuple[int, str, str]], context: dict) -> Optional[Tuple[str, str, str]]:
        """Propose a fix. LLM first (if wired); deterministic fallback otherwise/on failure."""
        if self.llm_client is not None:
            out = self._llm_fix(code, error_kind, error, history, context)
            if out is not None:
                return out
            # LLM unavailable/garbage this turn -> degrade honestly to the deterministic path,
            # do not fabricate. (Same contract as LLMProposer returning [].)
        return self._deterministic_fix(code, error_kind, error)

    # -- LLM backend -------------------------------------------------------------------
    def _prompt(self, code: str, error_kind: str, error: str,
                history: List[Tuple[int, str, str]], context: dict) -> str:
        hist = "\n".join(f"  attempt {a}: [{ek}] {msg}" for a, ek, msg in history[-6:]) \
            or "  (this is the first failure)"
        parts = [
            "You are debugging a scikit-learn pipeline. The code below defines "
            "build_estimator() and FAILED when fit/predicted in a sandbox.\n\n"
            f"Task kind: {context.get('task_kind')}. Features: {context.get('n_features')}.\n"
            f"Current error: [{error_kind}] {error}\n"
            "Earlier failed attempts (do NOT repeat these mistakes):\n"
            f"{hist}\n\n",
        ]
        # Inject diagnosis-aware repair context
        repair_strategies = context.get("repair_strategies", [])
        if repair_strategies:
            parts.append(
                "TARGETED REPAIR STRATEGIES:\n"
                + "\n".join(f"  - {s}" for s in repair_strategies[:3])
                + "\n\n"
            )
        llm_guidance = context.get("llm_guidance", "")
        if llm_guidance:
            parts.append(f"EXPERIMENT CONTEXT:\n{llm_guidance[:300]}\n\n")
        parts.append(
            "--- current code ---\n"
            f"{code}\n"
            "--- end code ---\n\n"
            "Return ONLY corrected Python code defining build_estimator() that returns an "
            "unfitted sklearn-compatible estimator. No prints, no markdown fences, no fit()."
        )
        return "".join(parts)

    def _llm_fix(self, code, error_kind, error, history, context) -> Optional[Tuple[str, str, str]]:
        try:
            out = self.llm_client(self._prompt(code, error_kind, error, history, context))
        except Exception:
            return None
        out = _strip_fences(out or "")
        if not out or "build_estimator" not in out or out.strip() == code.strip():
            return None
        return (out, "llm", "llm rewrite from typed error + history")

    # -- deterministic backend ---------------------------------------------------------
    def _deterministic_fix(self, code: str, error_kind: str,
                           error: str) -> Optional[Tuple[str, str, str]]:
        """Conservative, well-understood source transforms keyed off the error taxonomy.

        Each rule fixes a SPECIFIC, common failure mode and changes the source in a way a
        human reviewer would sign off on. Rules are tried in priority order; the first that
        produces a *different* string wins. Returns None if nothing applies (loop stops).
        """
        for rule in (
            self._fix_missing_build_estimator,
            self._fix_bad_import,
            self._fix_name_error,
            self._fix_unsupported_kwarg,
        ):
            new = rule(code, error_kind, error)
            if new is not None and new.strip() != code.strip():
                return (new, "deterministic", rule.__name__.replace("_fix_", "fix:"))
        return None

    # build error: "no callable build_estimator()" -> wrap the last defined estimator, or
    # synthesize a safe default matching nothing (we cannot guess intent, so a clear default).
    @staticmethod
    def _fix_missing_build_estimator(code: str, error_kind: str, error: str) -> Optional[str]:
        if error_kind != "build" or "build_estimator" not in error:
            return None
        if re.search(r"def\s+build_estimator\s*\(", code):
            return None  # it exists; this isn't the missing-function case
        # If the module already imports/constructs a known estimator, wrap it; else default.
        m = re.search(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Z]\w*\(.*\))\s*$", code, re.MULTILINE)
        if m:
            return code + f"\n\ndef build_estimator():\n    return {m.group(1)}\n"
        return (
            "from sklearn.ensemble import HistGradientBoostingClassifier\n\n"
            "def build_estimator():\n"
            "    # synthesized default (original code defined no build_estimator())\n"
            "    return HistGradientBoostingClassifier(random_state=0)\n"
        )

    # import error: the sandbox returns the ImportError message; rewrite the offending import.
    @staticmethod
    def _fix_bad_import(code: str, error_kind: str, error: str) -> Optional[str]:
        if error_kind not in ("import", "build"):
            return None
        # "cannot import name 'X' from 'pkg'" or "No module named 'pkg'"
        sym = re.search(r"cannot import name '([^']+)'", error)
        if sym:
            name = sym.group(1)
            fixed = _CANON_IMPORTS.get(name)
            if fixed:
                # replace any 'from ... import ... <name> ...' line that mentions the symbol
                pat = re.compile(r"^from\s+[\w.]+\s+import\s+.*\b" + re.escape(name) + r"\b.*$",
                                 re.MULTILINE)
                if pat.search(code):
                    return pat.sub(fixed, code, count=1)
        mod = re.search(r"No module named '([^']+)'", error)
        if mod:
            broken = mod.group(1).split(".")[0]
            # drop any import line referencing a non-existent top-level module; the rest of the
            # pipeline can still build if that import was decorative/unused.
            pat = re.compile(r"^(?:import|from)\s+" + re.escape(broken) + r"\b.*$", re.MULTILINE)
            if pat.search(code):
                return pat.sub("", code)
        return None

    # NameError surfaced as a fit/build error: "name 'X' is not defined" -> add the import.
    @staticmethod
    def _fix_name_error(code: str, error_kind: str, error: str) -> Optional[str]:
        m = re.search(r"name '([^']+)' is not defined", error)
        if not m:
            return None
        name = m.group(1)
        imp = _NAME_TO_IMPORT.get(name)
        if not imp:
            return None
        if imp in code:
            return None
        # prepend the missing import (idempotent: guarded by the `imp in code` check above)
        return imp + "\n" + code

    # fit/build error from an unsupported estimator kwarg:
    #   "__init__() got an unexpected keyword argument 'foo'"  /  "got an unexpected keyword..."
    @staticmethod
    def _fix_unsupported_kwarg(code: str, error_kind: str, error: str) -> Optional[str]:
        m = re.search(r"unexpected keyword argument '([^']+)'", error)
        if not m:
            return None
        kw = m.group(1)
        # remove `kw=<value>` from any constructor call. Handle a trailing or leading comma.
        # value is a balanced-ish token: number / string / simple identifier / call.
        val = r"(?:[^,()]+|\([^()]*\))+"
        patterns = [
            re.compile(r",\s*" + re.escape(kw) + r"\s*=\s*" + val),   # ", kw=..."
            re.compile(re.escape(kw) + r"\s*=\s*" + val + r"\s*,\s*"),  # "kw=..., "
            re.compile(re.escape(kw) + r"\s*=\s*" + val),             # "kw=..." alone
        ]
        new = code
        for pat in patterns:
            if pat.search(new):
                new = pat.sub("", new, count=1)
                break
        return new if new != code else None


# --------------------------------------------------------------------------- the loop

def repair_loop(program: Program, X_train, y_train, X_eval, *, kind: str,
                llm_client: Optional[Callable[[str], str]] = None, k: int = 3,
                wall_seconds: float = 60.0, cpu_seconds: int = 55,
                context: Optional[dict] = None,
                initial: Optional[RunResult] = None) -> RepairResult:
    """Run -> diagnose -> fix -> re-run, up to `k` repair attempts, threading error history.

    Args:
      program: the candidate to get running.
      X_train, y_train, X_eval: arrays the sandbox fits/predicts on. X_eval is a PROBE used
        only to confirm "does it run + produce well-shaped predictions"; it is NOT scored here
        and is NOT the sealed test. (The engine scores on VAL and certifies on SEALED itself.)
      kind: "classification" | "regression".
      llm_client: pluggable repairer backend; None -> deterministic heuristics only.
      k: max repair attempts after the initial run (k=0 just runs once, no repair).
      context: optional diagnosis dict (task_kind, n_features, ...) for the LLM prompt.
      initial: if you ALREADY ran `program` and have its RunResult, pass it to skip a re-run.

    Returns a RepairResult. `repaired=True` iff a fix turned a failing program into one that
    runs. On total failure the last attempted program + its typed RunResult are returned, so
    the caller records an HONEST failure -- never a fabricated success.
    """
    context = context or {}
    repairer = CodeRepairer(llm_client)
    steps: List[RepairStep] = []

    # attempt 0: the original (run it unless the caller already did).
    cur = program
    res = initial if (initial is not None and initial.program_id == program.id) else \
        sandbox.run_program(cur, X_train, y_train, X_eval, kind=kind,
                            wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
    steps.append(RepairStep(0, "original", "" if res.ok else res.error_kind,
                            "" if res.ok else res.error, res.ok, note="initial run"))
    if res.ok:
        return RepairResult(cur, res, repaired=False, steps=steps)

    # repair iterations 1..k
    for attempt in range(1, k + 1):
        history = [(s.attempt, s.error_kind, s.error) for s in steps if not s.ok]
        fix = repairer.fix(cur.code, res.error_kind, res.error, history, context)
        if fix is None:
            # nothing more to try (no rule applied / LLM gave no new candidate). Stop honestly.
            steps.append(RepairStep(attempt, "n/a", res.error_kind,
                                    "repairer produced no new candidate", ok=False,
                                    note="give up"))
            break
        new_code, src, note = fix
        cur = Program(
            code=new_code,
            source=program.source,            # keep origin; repair is a transform, not a new origin
            label=f"{program.label}~r{attempt}",
            parent_id=program.id,
            provenance={**dict(program.provenance),
                        "repair": {"attempt": attempt, "via": src, "note": note,
                                   "fixed_error_kind": res.error_kind}},
        )
        res = sandbox.run_program(cur, X_train, y_train, X_eval, kind=kind,
                                 wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
        steps.append(RepairStep(attempt, src, "" if res.ok else res.error_kind,
                                "" if res.ok else res.error, res.ok, note=note))
        if res.ok:
            return RepairResult(cur, res, repaired=True, steps=steps)

    return RepairResult(cur, res, repaired=False, steps=steps)


# --------------------------------------------------------------------------- proposer

class AgenticProposer:
    """A ProposalSource that authors candidate code and SELF-REPAIRS it before returning.

    Two modes:
      * author-from-scratch (default): builds candidate code from the round's diagnosis
        context (the LLM writes it if a client is wired; otherwise a deterministic, valid
        baseline is authored so the source is never empty), then runs each through
        repair_loop so what we hand the engine already executes.
      * wrap a base proposer: AgenticProposer(base=MutationProposer()) makes that proposer
        agentic by repairing whatever it emits.

    `propose(context)` returns list[Program] -- the SAME contract as the other proposers, so
    it drops straight into ResearchEngine(proposers=[...]). It returns repaired (running)
    programs when repair succeeds, and the best-effort attempt (still typed-failing) otherwise
    so the engine logs an honest failure rather than a fabricated success.
    """

    def __init__(self, base: Optional[object] = None,
                 llm_client: Optional[Callable[[str], str]] = None,
                 n_author: int = 1, k: int = 3,
                 wall_seconds: float = 60.0, cpu_seconds: int = 55):
        self.base = base
        self.llm_client = llm_client
        self.n_author = max(0, int(n_author))
        self.k = int(k)
        self.wall_seconds = wall_seconds
        self.cpu_seconds = cpu_seconds

    # -- candidate authoring -----------------------------------------------------------
    def _author_from_scratch(self, context: dict) -> List[Program]:
        """Write candidate code from the diagnosis context.

        With an LLM, ask it (open-ended). Without, author a deterministic, valid baseline so
        the loop has something to run -- this is a fallback floor, never a promoter.
        """
        kind = context.get("task_kind", "classification")
        out: List[Program] = []
        if self.llm_client is not None:
            prompt = _author_prompt(context)
            for i in range(self.n_author):
                try:
                    code = _strip_fences(self.llm_client(prompt) or "")
                except Exception:
                    break
                if code and "build_estimator" in code:
                    out.append(Program(code=code, source="agentic",
                                       label=f"agentic{context.get('round', 0)}_{i}",
                                       provenance={"authored": "llm"}))
        if not out:
            # honest fallback: a known-valid baseline matched to the kind.
            out.append(Program(code=_baseline_code(kind), source="agentic",
                               label=f"agentic_base_{kind}",
                               provenance={"authored": "deterministic_fallback"}))
        return out

    # -- ProposalSource API ------------------------------------------------------------
    def propose(self, context: dict) -> List[Program]:
        """Author (or wrap) candidates, repair each in the sandbox, return running code."""
        tried = set(context.get("tried_labels", ()))
        if self.base is not None and hasattr(self.base, "propose"):
            candidates = list(self.base.propose(context))
        else:
            candidates = self._author_from_scratch(context)

        kind = context.get("task_kind", "classification")
        Xtr, ytr, Xpr = _resolve_probe(context, kind)

        out: List[Program] = []
        for cand in candidates:
            if cand.label in tried:
                continue
            rr = repair_loop(cand, Xtr, ytr, Xpr, kind=kind, llm_client=self.llm_client,
                             k=self.k, wall_seconds=self.wall_seconds,
                             cpu_seconds=self.cpu_seconds, context=context)
            # Return the (possibly repaired) program. The engine re-runs it on the real VAL
            # split; a repaired program runs cleanly there. If repair failed, we still return
            # the attempt so the engine records the typed failure honestly.
            out.append(rr.program)
        return out


# --------------------------------------------------------------------------- helpers

def attach_probe_to_context(context: dict, X_train, y_train, X_probe, kind: str) -> dict:
    """Optionally let the integrator give AgenticProposer the REAL train/probe arrays.

    If the engine is extended to add these keys to the per-round context, repair iterates
    against the actual data instead of a synthetic probe. Returns a NEW dict (no mutation of
    the caller's context). Purely additive: absent these keys, _resolve_probe synthesizes a
    matched probe and the loop still works.
    """
    new = dict(context)
    new.update({"X_train": np.asarray(X_train, dtype=float),
                "y_train": np.asarray(y_train),
                "X_probe": np.asarray(X_probe, dtype=float),
                "kind": kind})
    return new


def _resolve_probe(context: dict, kind: str):
    """Get (X_train, y_train, X_eval) for the repair loop.

    Prefer real arrays the integrator attached; otherwise synthesize a tiny, well-conditioned
    probe matched to kind + n_features so repair can verify "does it run + predict the right
    shape". The probe is NEVER scored and NEVER the sealed test -- it only drives iteration.
    """
    if all(k in context for k in ("X_train", "y_train", "X_probe")):
        return context["X_train"], context["y_train"], context["X_probe"]
    n_features = int(context.get("n_features") or 8)
    n = max(40, 4 * n_features)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n_features))
    if kind == "classification":
        # a learnable 2-class signal so estimators that need >1 class in train don't error
        y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int).astype(str)
    else:
        y = X[:, 0] + 0.1 * rng.standard_normal(n)
    n_probe = max(8, n // 5)
    Xpr = rng.standard_normal((n_probe, n_features))
    return X, y, Xpr


def _author_prompt(context: dict) -> str:
    kind = context.get("task_kind", "classification")
    errs = context.get("recent_errors", [])
    err_txt = "\n".join(f"  - {lab}: [{ek}] {msg}" for lab, ek, msg in errs[:5]) or "  (none yet)"
    return (
        "You are an ML engineer writing a scikit-learn pipeline from scratch.\n"
        f"Task kind: {kind}. Features: {context.get('n_features')}. "
        f"Train size: {context.get('n_train')}.\n"
        f"Best so far: {context.get('best_label')} (val {context.get('best_score')}).\n"
        "Recent failed attempts to avoid:\n"
        f"{err_txt}\n\n"
        "Return ONLY Python code defining build_estimator() returning an unfitted "
        "sklearn-compatible estimator. Feature engineering / transforms allowed. "
        "No fit, no prints, no markdown fences."
    )


def _baseline_code(kind: str) -> str:
    """A known-valid baseline so author-from-scratch never returns empty (fallback floor)."""
    if kind == "regression":
        return (
            "from sklearn.ensemble import HistGradientBoostingRegressor\n\n"
            "def build_estimator():\n"
            "    return HistGradientBoostingRegressor(random_state=0)\n"
        )
    return (
        "from sklearn.ensemble import HistGradientBoostingClassifier\n\n"
        "def build_estimator():\n"
        "    return HistGradientBoostingClassifier(random_state=0)\n"
    )


def _strip_fences(text: str) -> str:
    """Strip ```python ... ``` fences an LLM may wrap code in; return inner code."""
    t = text.strip()
    if t.startswith("```"):
        # drop first fence line and a trailing fence if present
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


# Canonical replacements for common WRONG sklearn import lines (symbol -> correct full line).
# These are the high-frequency mistakes an LLM or a careless author makes (right name, wrong
# module). Conservative + auditable; extend as new misfires are observed.
_CANON_IMPORTS = {
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
    "HistGradientBoostingClassifier": "from sklearn.ensemble import HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor": "from sklearn.ensemble import HistGradientBoostingRegressor",
    "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
    "Ridge": "from sklearn.linear_model import Ridge",
    "Lasso": "from sklearn.linear_model import Lasso",
    "LinearRegression": "from sklearn.linear_model import LinearRegression",
    "SVC": "from sklearn.svm import SVC",
    "SVR": "from sklearn.svm import SVR",
    "StandardScaler": "from sklearn.preprocessing import StandardScaler",
    "PolynomialFeatures": "from sklearn.preprocessing import PolynomialFeatures",
    "Pipeline": "from sklearn.pipeline import Pipeline",
    "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
    "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
}

# For NameError repair: a bare name used but never imported -> the import that defines it.
_NAME_TO_IMPORT = dict(_CANON_IMPORTS)
_NAME_TO_IMPORT.update({
    "np": "import numpy as np",
    "make_pipeline": "from sklearn.pipeline import make_pipeline",
    "ColumnTransformer": "from sklearn.compose import ColumnTransformer",
    "TransformedTargetRegressor": "from sklearn.compose import TransformedTargetRegressor",
})
