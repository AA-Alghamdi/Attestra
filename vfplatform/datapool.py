"""DATA POOL -- a curated, hygiene-certified registry of datasets the engine learns across.

STATUS (2026-06): WIRED into the REGENERATIVE researcher as the DATA-HYGIENE GATE. When constructed with
data_pool_root, RecipeResearcher.run() admits the arena's dataset into this append-only pool BEFORE any
search -- DataPool.add(require_cert=True) runs data_cert.certify_dataset, so a dataset with near-duplicate
train/sealed straddle (the non-negotiable leak) is REFUSED outright (cert.refused=True, peeks_used=0). This
is the data-side complement to the meta-certifier's framing-side gate. Falsified by a hermetic lock
(test_recipe_research.py::test_datapool_gate_refuses_contaminated_dataset) and run on real WILDS Camelyon17
(dataset_name="wilds_camelyon17"). The cross-problem MEMORY roles below remain future work.

WHY THIS EXISTS
---------------
The compounding edge an average researcher cannot match comes from MEMORY ACROSS PROBLEMS. The data pool
is the species-level memory: a registry of datasets (across task types/modalities), each admitted only
after passing a DATA CERTIFICATE (data_cert.certify_dataset), each carrying a fingerprint, declared
leakage hazards, and a frozen baseline. It serves three roles:
  1. meta-learning fuel  -- a corpus of (fingerprint, operator, outcome) the surrogate/proposer learn from.
  2. replication arena    -- the D2..Dk a method must replicate on to earn a GENERAL (not dataset-specific)
                             certificate (see replication.py).
  3. overfitting guard    -- a held-out suite to measure real progress, not benchmark-hacking.

ADMISSION IS GATED. `add` refuses a dataset whose data certificate fails (e.g. train/sealed near-duplicate
leakage). This keeps the pool's compounding trustworthy: a poisoned dataset cannot silently corrupt every
cross-dataset claim.

CONTRACT: metadata registry persisted as append-only JSONL (last write per name wins). It stores the
certificate + fingerprint + hazards, NOT raw data (the loader is the caller's). No model certifier here.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from . import data_cert


@dataclass
class PoolDataset:
    name: str
    modality: str
    task_type: str
    n: int
    fingerprint: dict = field(default_factory=dict)
    leakage_hazards: List[str] = field(default_factory=list)
    data_certificate: dict = field(default_factory=dict)
    baseline: Optional[dict] = None
    split_method: Optional[str] = None
    source: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


class DataPoolError(ValueError):
    pass


class DataPool:
    """Append-only registry of admitted datasets. `root` is a directory; the manifest is manifest.jsonl."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        self._by_name: Dict[str, PoolDataset] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self._by_name[d["name"]] = PoolDataset(**d)

    def _append(self, ds: PoolDataset) -> None:
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps(ds.as_dict(), sort_keys=True) + "\n")

    def add(self, name: str, *, modality: str, task_type: str,
            X=None, y=None, train_idx: Optional[Sequence[int]] = None,
            sealed_idx: Optional[Sequence[int]] = None, fingerprint: Optional[dict] = None,
            leakage_hazards: Optional[List[str]] = None, baseline: Optional[dict] = None,
            split_method: Optional[str] = None, source: Optional[str] = None,
            require_cert: bool = True, **cert_kwargs) -> PoolDataset:
        """Admit a dataset. If X/y/train_idx/sealed_idx are given, the data certificate is computed and
        (when require_cert) admission FAILS if it does not pass. fingerprint may be supplied or computed
        from the data via casebase_store.fingerprint."""
        cert = {}
        if X is not None and y is not None and train_idx is not None and sealed_idx is not None:
            report = data_cert.certify_dataset(X, y, train_idx=train_idx, sealed_idx=sealed_idx,
                                               **cert_kwargs)
            cert = report.as_dict()
            if require_cert and not report.passed:
                raise DataPoolError(
                    f"dataset {name!r} REFUSED: data certificate failed -> {report.warnings}. "
                    "Fix the split/data before admitting it; a leaky dataset poisons the pool.")
        if fingerprint is None and X is not None:
            try:
                from . import casebase_store
                fingerprint = casebase_store.fingerprint(X, kind=modality, task_type=task_type)
            except Exception:  # noqa: BLE001  fingerprint is best-effort metadata, never blocks admission
                fingerprint = {}
        n = int(len(y)) if y is not None else 0
        ds = PoolDataset(name=name, modality=modality, task_type=task_type, n=n,
                         fingerprint=fingerprint or {}, leakage_hazards=list(leakage_hazards or []),
                         data_certificate=cert, baseline=baseline, split_method=split_method, source=source)
        self._by_name[name] = ds
        self._append(ds)
        return ds

    def get(self, name: str) -> Optional[PoolDataset]:
        return self._by_name.get(name)

    def names(self) -> List[str]:
        return sorted(self._by_name)

    def query(self, *, modality: Optional[str] = None,
              task_type: Optional[str] = None) -> List[PoolDataset]:
        out = []
        for ds in self._by_name.values():
            if modality is not None and ds.modality != modality:
                continue
            if task_type is not None and ds.task_type != task_type:
                continue
            out.append(ds)
        return sorted(out, key=lambda d: d.name)

    def replication_set(self, exclude: str, *, modality: Optional[str] = None,
                        task_type: Optional[str] = None) -> List[PoolDataset]:
        """The D2..Dk a method must replicate on for a GENERAL certificate: same modality+task_type as the
        anchor dataset, excluding the anchor itself."""
        anchor = self._by_name.get(exclude)
        mod = modality if modality is not None else (anchor.modality if anchor else None)
        tt = task_type if task_type is not None else (anchor.task_type if anchor else None)
        return [d for d in self.query(modality=mod, task_type=tt) if d.name != exclude]

    def counts(self) -> dict:
        by_mod: Dict[str, int] = {}
        for ds in self._by_name.values():
            by_mod[ds.modality] = by_mod.get(ds.modality, 0) + 1
        return {"total": len(self._by_name), "by_modality": by_mod}


__all__ = ["DataPool", "PoolDataset", "DataPoolError"]
