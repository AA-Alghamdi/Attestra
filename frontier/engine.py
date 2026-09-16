"""The corrected generative research loop (Phase 0 spine).

One round:  propose -> sandbox-execute -> score on VAL -> select -> (carry diagnosis).
After rounds: certify the single VAL winner on the HELD-OUT SEALED test (one counted peek).

Integrity invariants this loop enforces (the audit found each broken elsewhere):
  - Selection touches the validation split only; the sealed test is read exactly once,
    at the end, for the winner -- never during search. (PR19 certified on val.)
  - Candidate code runs in a real subprocess sandbox, never in-process exec. (PR18/PR19.)
  - The sandbox returns predictions only; every number is computed here via science.py.
  - The selected winner IS the thing certified -- generated or seed, same gate. (PR18 emitted
    the catalog's certificate when the generative path "won".)
  - Round N's diagnosis (champion + typed failures) conditions round N+1's proposals.
    (loop.py:781 discarded `_relevant`; engine.py:395 consumed only `gap`.)

Outcome is honest: if no candidate clears theta on the sealed lower bound, the result is a
decline carrying the best certificate attempt, not a relabeled val score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import certify
from . import sandbox
from .program import Program
from .proposers import SeedProposer, MutationProposer, LLMProposer, ProposalSource
from .task import Task


@dataclass
class EngineConfig:
    rounds: int = 3
    seed: int = 0
    test_frac: float = 0.30
    val_frac: float = 0.20
    wall_seconds: float = 60.0
    cpu_seconds: int = 55
    llm_client: Optional[Callable[[str], str]] = None   # prompt -> code; None => LLM path inactive


@dataclass
class _Record:
    program_id: str
    label: str
    source: str
    ok: bool
    val_score: Optional[float] = None
    error_kind: str = ""
    error: str = ""
    wall_seconds: float = 0.0


@dataclass
class EngineResult:
    certified: bool
    certificate: Optional[dict]          # the sealed-test certificate dict (or None on hard decline)
    winner: Optional[Program]
    winner_val_score: Optional[float]
    history: List[_Record] = field(default_factory=list)
    diagnosis_trail: List[dict] = field(default_factory=list)
    split_meta: dict = field(default_factory=dict)
    decline_reason: str = ""
    llm_active: bool = False

    def summary(self) -> str:
        lines = [f"task certified={self.certified}"]
        if self.winner is not None:
            lines.append(f"winner={self.winner.label} ({self.winner.source}) "
                         f"val={self.winner_val_score:.4f}")
        if self.certificate is not None:
            c = self.certificate
            lines.append(f"sealed: observed={c.get('observed')} lower_bound={c.get('lower_bound')} "
                         f"theta={c.get('theta')} peeks={c.get('peeks')} -> certified={c.get('certified')}")
        if self.decline_reason:
            lines.append(f"decline: {self.decline_reason}")
        lines.append(f"candidates tried={len(self.history)} "
                     f"(ok={sum(1 for r in self.history if r.ok)}), llm_active={self.llm_active}")
        return "\n".join(lines)


class ResearchEngine:
    def __init__(self, config: Optional[EngineConfig] = None,
                 proposers: Optional[List[ProposalSource]] = None):
        self.cfg = config or EngineConfig()
        if proposers is None:
            proposers = [SeedProposer(), MutationProposer(), LLMProposer(self.cfg.llm_client)]
        self.proposers = proposers

    def run(self, task: Task) -> EngineResult:
        cfg = self.cfg
        splits = certify.make_splits(task, seed=cfg.seed, test_frac=cfg.test_frac,
                                     val_frac=cfg.val_frac)
        X_train = Task.rows_to_X(splits.train_rows)
        y_train = Task.rows_to_y(splits.train_rows, task.kind)
        X_val = Task.rows_to_X(splits.val_rows)

        history: List[_Record] = []
        trail: List[dict] = []
        tried_labels: set = set()
        recent_errors: List[tuple] = []
        llm_active = any(isinstance(p, LLMProposer) and p.client is not None for p in self.proposers)

        best_score: Optional[float] = None
        best_prog: Optional[Program] = None

        for r in range(cfg.rounds):
            context = {
                "task_kind": task.kind,
                "n_features": task.n_features,
                "n_train": len(splits.train_rows),
                "round": r,
                "tried_labels": set(tried_labels),
                "best_label": best_prog.label if best_prog else None,
                "best_score": round(best_score, 4) if best_score is not None else None,
                "best_id": best_prog.id if best_prog else None,
                "best_recipe": (best_prog.provenance.get("recipe") if best_prog else None),
                "recent_errors": list(recent_errors),
            }

            # gather + dedup proposals for this round
            proposals: List[Program] = []
            seen_ids = set()
            for src in self.proposers:
                for p in src.propose(context):
                    if p.label in tried_labels or p.id in seen_ids:
                        continue
                    seen_ids.add(p.id)
                    proposals.append(p)

            round_log = {"round": r, "n_proposals": len(proposals),
                         "best_in_label": context["best_label"], "best_in_score": context["best_score"]}
            if not proposals:
                round_log["note"] = "no new proposals"
                trail.append(round_log)
                break

            for p in proposals:
                res = sandbox.run_program(p, X_train, y_train, X_val, kind=task.kind,
                                          wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
                tried_labels.add(p.label)
                rec = _Record(p.id, p.label, p.source, res.ok, error_kind=res.error_kind,
                              error=res.error, wall_seconds=round(res.wall_seconds, 2))
                if res.ok:
                    score = certify.score_val(task, splits.val_rows, res.preds)
                    rec.val_score = round(score, 4)
                    if best_score is None or score > best_score:
                        best_score, best_prog = score, p
                else:
                    recent_errors.append((p.label, res.error_kind, res.error))
                history.append(rec)

            round_log["best_out_label"] = best_prog.label if best_prog else None
            round_log["best_out_score"] = round(best_score, 4) if best_score is not None else None
            trail.append(round_log)

        # ---- no candidate ran at all -> hard, honest decline
        if best_prog is None:
            return EngineResult(False, None, None, None, history, trail, splits.meta,
                                decline_reason="no candidate executed successfully", llm_active=llm_active)

        # ---- certify the single winner on the sealed test (the only time sealed is touched)
        X_sealed = Task.rows_to_X(splits.sealed_rows)
        final = sandbox.run_program(best_prog, X_train, y_train, X_sealed, kind=task.kind,
                                    wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
        if not final.ok:
            return EngineResult(False, None, best_prog, round(best_score, 4), history, trail,
                                splits.meta,
                                decline_reason=f"winner failed on sealed re-fit: [{final.error_kind}] {final.error}",
                                llm_active=llm_active)

        cert = certify.certify_on_sealed(task, splits, final.preds)
        return EngineResult(
            certified=bool(cert.get("certified")),
            certificate=cert,
            winner=best_prog,
            winner_val_score=round(best_score, 4),
            history=history,
            diagnosis_trail=trail,
            split_meta=splits.meta,
            decline_reason=("" if cert.get("certified")
                            else "sealed lower bound does not clear theta (honest decline)"),
            llm_active=llm_active,
        )
