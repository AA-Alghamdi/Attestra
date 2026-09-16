"""ResearchMemory -- the durable learning substrate that makes the loop get better every run.

WHAT THIS IS
------------
The recursive loop (``loop.run_goal_loop``) ranks its bounded move catalog by Value-of-Information
(``voi.voi_rank`` -> expected_gain / cost). Out of the box VoI uses each move's STATIC catalog prior
until the IN-RUN ``voi.CaseBase`` of realized gains replaces it -- but that case-base is forgotten the
moment the run ends. So Attestra never carried what it learned from one experiment into the next, and
never transferred "on data LIKE this, family X paid off" to a *new* dataset.

``ResearchMemory`` closes that gap. It is a drop-in ``voi.CaseBase`` (same ``record`` / ``record_cost`` /
``expected_gain`` / ``expected_cost`` surface, so ``voi_rank`` consumes it unchanged) wrapped around the
DURABLE, dataset-aware ``casebase_store.CaseBaseStore``:

  * at construction it computes the dataset FINGERPRINT (meta-features: modality, size, #features,
    #classes, balance) and asks the store for a nearest-neighbour WARM START -- per-family realized
    (gain, cost) aggregated from the most similar PAST datasets, plus a negative-memory "avoid" list;
  * ``expected_gain`` / ``expected_cost`` consult three tiers, strongest first:
        1. this run's realized history (the base ``CaseBase`` behaviour -- always wins once observed),
        2. the cross-dataset warm-start prior for the move's FAMILY (so a brand-new dataset starts from
           informed priors, and known dead ends sink below any cold family),
        3. the move's static catalog prior (cold start -- byte-identical to the pre-memory ranking);
  * ``observe(family, gain, cost)`` appends the realized outcome back to the durable store, tagged with
    the compute ``device`` ("cpu" / "cuda") -- so a GPU campaign's results accumulate and compound.

INTEGRITY CONTRACT (hard boundary)
----------------------------------
This is PURE PROPOSAL POLICY, exactly like ``casebase_store``: it only ever REORDERS the bounded catalog
the proposer already draws from. It NEVER reads the sealed test, touches a certificate / theta, or mutates
a frozen file (science.py / sealed.py). With no store it degenerates to a plain ``voi.CaseBase`` (the
default, cold-start path stays byte-identical).
"""
from __future__ import annotations

from typing import Optional

from . import casebase_store
from .voi import CaseBase

# A warm-start prior is never allowed to read as exactly 0 for a family with positive cross-dataset
# evidence (it must still out-rank a never-seen family); a KNOWN dead end is pushed strictly below the
# faintest cold prior so it sinks to the bottom of the VoI order without being hard-banned.
_FAINT = 1e-4


def family_of(move) -> str:
    """The FAMILY a move belongs to. Grid-proposed moves are named ``"family|param=val|..."`` and the loop
    itself keys family-diversity on ``name.split("|")[0]``; static harness moves carry a plain family-ish
    name. We use the same split so a recorded outcome and a warm-start lookup agree on the key."""
    nm = getattr(move, "name", "") or ""
    return nm.split("|")[0]


class ResearchMemory(CaseBase):
    """Durable, dataset-aware ``voi.CaseBase``. See module docstring.

    Construct via :meth:`build` (it computes the fingerprint + warm start for you). A bare constructor with
    ``store=None`` is a plain in-run ``CaseBase`` -- handy for tests and the cold-start default.
    """

    def __init__(self, run_casebase_path: Optional[str] = None, *,
                 store: Optional["casebase_store.CaseBaseStore"] = None,
                 fingerprint: Optional[dict] = None, warm: Optional[dict] = None,
                 device: str = "cpu"):
        super().__init__(run_casebase_path)
        self.store = store
        self.fp = fingerprint
        warm = warm or {}
        self.warm_families = dict(warm.get("families", {}))
        self.avoid = set(warm.get("avoid", []))
        self.n_neighbors = int(warm.get("n_neighbors", 0))
        self.device = device or "cpu"

    @classmethod
    def build(cls, profile: dict, *, store: Optional["casebase_store.CaseBaseStore"],
              run_casebase_path: Optional[str] = None, kind: Optional[str] = None,
              task_type: Optional[str] = None, device: str = "cpu",
              k: int = 12, max_distance: float = 6.0) -> "ResearchMemory":
        """Fingerprint ``profile`` (the loop's ``_profile`` dict or a records list), pull the nearest-
        neighbour warm start from ``store``, and return a ready ResearchMemory. Best-effort: any failure in
        the (additive) memory layer yields a cold-start ResearchMemory rather than breaking the run."""
        fp = warm = None
        if store is not None:
            try:
                fp = casebase_store.fingerprint(profile, kind=kind, task_type=task_type)
                warm = store.warm_start(fp, k=k, max_distance=max_distance)
            except Exception:  # noqa: BLE001  memory is additive; never break a run
                fp = warm = None
        return cls(run_casebase_path, store=store, fingerprint=fp, warm=warm, device=device)

    # --- VoI surface (three-tier: realized -> cross-dataset warm prior -> catalog prior) -------------
    def expected_gain(self, move):
        hist = self.gains.get(move.name, [])
        if hist:                                              # (1) in-run realized dominates
            return max(0.0, sum(hist) / len(hist))
        fam = family_of(move)
        wf = self.warm_families.get(fam)
        if wf is None:                                        # (3) no cross-run signal -> catalog prior
            return move.prior_gain
        if fam in self.avoid:                                 # known dead end -> below any cold family
            return min(move.prior_gain * 0.1, _FAINT)
        return max(float(wf.get("prior_gain", 0.0)), _FAINT)  # (2) cross-dataset warm prior

    def expected_cost(self, move):
        hist = self.costs.get(move.name, [])
        if hist:
            return max(1e-6, sum(hist) / len(hist))
        fam = family_of(move)
        wf = self.warm_families.get(fam)
        if wf is not None:
            return max(1e-6, float(wf.get("prior_cost", move.prior_cost)))
        return max(move.prior_cost, 1e-6)

    # --- durable write side --------------------------------------------------------------------------
    def observe(self, family: str, gain: float, cost: float) -> None:
        """Append one realized (fingerprint, family, gain, cost, device) outcome to the durable store so
        future runs -- on this dataset OR a similar one -- warm-start from it. No-op without a store/fp."""
        if self.store is None or self.fp is None or not family:
            return
        try:
            self.store.record_outcome(self.fp, family, gain, cost, device=self.device)
        except Exception:  # noqa: BLE001  the durable write is additive; never break a run
            pass

    # --- introspection (reporting / tests) -----------------------------------------------------------
    def warm_summary(self) -> dict:
        """What this run inherited from prior experiments: neighbours found, families with a learned prior,
        and the dead ends it will avoid. Pure reporting."""
        return {
            "fp_key": (self.fp or {}).get("key"),
            "n_neighbors": self.n_neighbors,
            "warm_families": sorted(self.warm_families),
            "avoid": sorted(self.avoid),
            "device": self.device,
        }
