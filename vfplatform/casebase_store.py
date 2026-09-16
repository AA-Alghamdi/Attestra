"""Durable, warm-started proposal-policy memory: make the search cycle smarter every run.

WHAT THIS IS
------------
Today the recursive cycle (run_goal_loop) sweeps a fixed catalog, FORGETS every run, and
re-sweeps from scratch on every plateau. The in-process ``voi.CaseBase`` warm-starts VoI
from realized gains, but only with a SINGLE flat ``move_name -> [gains]`` table that is keyed
neither by the dataset nor by family, so "on data like THIS, hist_gbm gained a lot but knn
didn't" is never learned. This module is the missing durable, dataset-aware proposal memory:

  (a) DATASET FINGERPRINT     -- a small, comparable signature of a task (shapes + balance
                                 bucket + modality) computed from a records list OR a profile
                                 dict (the loop's ``_profile``), so the same dataset maps to
                                 the same fingerprint regardless of which entry point built it.
  (b) APPEND-ONLY OUTCOME LOG -- a JSONL under ``vf_runs/`` of realized
                                 (fingerprint, family, gain, cost) tuples, one per executed
                                 move-on-a-dataset. Atomic append; tolerant of corrupt lines.
  (c) warm_start(fingerprint) -- per-family PRIOR (mean gain, mean cost, count, win-rate)
                                 aggregated from the NEAREST past fingerprints (distance-
                                 weighted), so a brand-new task starts with informed priors
                                 instead of a flat grid sweep.
  (d) LEDGER MINING           -- mine an existing ``cross_experiment.PromotionLedger`` JSONL
                                 for which families beat their baseline on which profiles and
                                 fold those wins into the priors (best-effort: the frozen
                                 promotion-ledger schema does not always carry a family, so
                                 unattributable rows are skipped, never guessed).
  (e) NEGATIVE MEMORY         -- record (fingerprint, family/approach) that FAILED (zero/near-
                                 zero realized gain, or a futility/honest-stop outcome) so the
                                 proposer can DEPRIORITIZE dead ends on similar data.

INTEGRITY CONTRACT (this is a hard boundary, not a style note)
--------------------------------------------------------------
This module is PURE PROPOSAL POLICY. It only ever READS past outcomes and EMITS priors /
penalties that reorder or reweight the bounded catalog the proposer already draws from. It
NEVER:
  * touches a certificate, the sealed peek, select-then-bound, or any threshold theta;
  * imports ``vfplatform.loop`` (it ACCEPTS a profile/diagnosis/family-string, by design, so
    it can be unit-tested and wired without a circular import);
  * mutates the frozen files (science.py / sealed.py / harness.py / battery.py / frontdoor.py).

Self-contained: stdlib only (json, os, math, tempfile). No numpy, no torch, no sklearn.
NOT wired into the loop; the operator hand-wires it (see ``CaseBaseStore`` docstring + the
``__main__`` runnable example). Append semantics mirror ``cross_experiment.PromotionLedger``:
atomic tmp+fsync+os.replace, append-only, corrupt-tolerant.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- defaults
DEFAULT_OUTCOME_PATH = os.path.join("vf_runs", "_casebase_outcomes.jsonl")
DEFAULT_NEGATIVE_PATH = os.path.join("vf_runs", "_casebase_negative.jsonl")

# A realized gain at or below this counts as a "dead end" for negative memory. The val
# objective is on [0,1]-ish metrics (accuracy / R^2-ish), so ~1e-3 is "did not move the bound".
DEAD_END_GAIN = 1e-3


# =========================================================================== fingerprint
def _class_balance_bucket(n_classes: Optional[int], class_counts) -> str:
    """Coarse, comparable balance bucket. We bucket (rather than store the raw ratio) so two
    datasets that are 'similarly skewed' share a fingerprint coordinate and so warm-start can
    match them. Buckets: na | regression | balanced | mild | imbalanced | severe.

    Severity is the ratio of the largest to the smallest class count (the imbalance ratio, IR;
    a standard summary in the class-imbalance literature). IR<=1.5 balanced; <=3 mild; <=10
    imbalanced; else severe. With only ``n_classes`` and no counts we cannot judge balance, so
    we return 'na' rather than guessing 'balanced'."""
    if n_classes is None or (isinstance(n_classes, int) and n_classes <= 1):
        return "regression" if n_classes == 1 else "na"
    if not class_counts:
        return "na"
    counts = [c for c in class_counts if c and c > 0]
    if len(counts) < 2:
        return "na"
    ir = max(counts) / float(min(counts))
    if ir <= 1.5:
        return "balanced"
    if ir <= 3.0:
        return "mild"
    if ir <= 10.0:
        return "imbalanced"
    return "severe"


def _size_bucket(n: Optional[int]) -> str:
    """Log-scale size bucket so 1.1k and 1.3k rows match but 1k and 100k do not. Order-of-
    magnitude buckets are the natural granularity for 'data like this' (a 200-row task and a
    200k-row task want different families even at the same shape)."""
    if not n or n <= 0:
        return "na"
    e = int(math.floor(math.log10(n)))
    return f"1e{e}"


def _modality(kind: Optional[str], task_type: Optional[str], n_features: Optional[int]) -> str:
    """Coarse modality label. We prefer the explicit ``kind`` (the loop sets 'tabular' /
    'text' / 'vision' / 'timeseries'); fall back to a width heuristic only when kind is absent
    (very wide -> likely text/vision bag-of-features; else tabular). The fallback is labeled a
    heuristic and never overrides an explicit kind."""
    if kind:
        k = str(kind).lower()
        if "text" in k:
            return "text"
        if "vision" in k or "image" in k:
            return "vision"
        if "time" in k or "series" in k:
            return "timeseries"
        if "tab" in k:
            return "tabular"
        return k
    # heuristic fallback (labeled): no kind available, infer from width
    if n_features and n_features >= 2000:
        return "wide_heuristic"
    return "tabular_heuristic"


def fingerprint(source, *, kind: Optional[str] = None, task_type: Optional[str] = None,
                target_key: str = "target") -> Dict:
    """Compute a DATASET FINGERPRINT from EITHER a records list OR a profile dict.

    Two entry points, one signature, so the loop's ``_profile`` and a raw ``records`` list map
    to the SAME fingerprint for the same dataset.

    Records-list path: ``source`` is a list of row dicts. We measure n_rows, n_features (max
    key-count over a sample of rows, excluding the target), n_classes + class counts (from
    ``target_key``; a continuous numeric target -> regression). ``kind`` may be passed to fix
    the modality (else heuristic).

    Profile-dict path: ``source`` is a dict carrying any of n_train/n_val/n_rows, n_features,
    n_classes, kind, task_type (exactly the loop's ``_profile`` shape). We do not have raw
    counts there, so the balance bucket is 'na' unless the profile carries ``class_counts``.

    Returns a dict with both the RAW coordinates (for transparency / mining) and a stable
    ``key`` string (the bucketed signature used for nearest-match warm-start).
    """
    n_rows: Optional[int] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    class_counts: Optional[List[int]] = None

    if isinstance(source, dict):
        prof = source
        kind = kind or prof.get("kind")
        task_type = task_type or prof.get("task_type")
        n_rows = prof.get("n_rows")
        if n_rows is None:
            ntr, nva = prof.get("n_train"), prof.get("n_val")
            if ntr is not None or nva is not None:
                n_rows = int(ntr or 0) + int(nva or 0)
        n_features = prof.get("n_features")
        n_classes = prof.get("n_classes")
        cc = prof.get("class_counts")
        if cc:
            class_counts = [int(c) for c in cc]
    elif isinstance(source, (list, tuple)):
        rows = list(source)
        n_rows = len(rows)
        if rows:
            sample = rows[: min(len(rows), 500)]
            feat_keys = set()
            for r in sample:
                if isinstance(r, dict):
                    feat_keys |= {k for k in r.keys() if k != target_key}
            n_features = len(feat_keys) if feat_keys else None
            tvals = [r.get(target_key) for r in rows if isinstance(r, dict) and target_key in r]
            tvals = [t for t in tvals if t is not None]
            if tvals:
                # classification iff the target is non-numeric OR has few distinct values
                distinct = set(map(_norm_label, tvals))
                numeric = all(_is_number(t) for t in tvals)
                if numeric and len(distinct) > min(20, max(2, len(tvals) // 10)):
                    n_classes = 1            # treat as regression
                else:
                    n_classes = len(distinct)
                    counts = {}
                    for t in tvals:
                        counts[_norm_label(t)] = counts.get(_norm_label(t), 0) + 1
                    class_counts = sorted(counts.values(), reverse=True)
    else:
        raise ValueError("fingerprint(source): source must be a records list or a profile dict")

    modality = _modality(kind, task_type, n_features)
    size_b = _size_bucket(n_rows)
    feat_b = _size_bucket(n_features)
    bal_b = _class_balance_bucket(n_classes, class_counts)
    nc_b = "na" if n_classes is None else ("reg" if n_classes == 1 else str(int(n_classes)))

    key = "|".join([modality, size_b, feat_b, nc_b, bal_b])
    return {
        "key": key,
        "modality": modality,
        "n_rows": n_rows,
        "n_features": n_features,
        "n_classes": n_classes,
        "class_counts": class_counts,
        "size_bucket": size_b,
        "feature_bucket": feat_b,
        "n_classes_bucket": nc_b,
        "balance_bucket": bal_b,
    }


def _is_number(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _norm_label(x):
    """Stable label key: numbers compare by value, everything else by string. Avoids 1 vs '1'
    being counted as two classes."""
    if isinstance(x, bool):
        return f"b:{x}"
    if isinstance(x, (int, float)):
        return f"n:{float(x)}"
    return f"s:{x}"


def fingerprint_distance(fp_a: Dict, fp_b: Dict) -> float:
    """Distance between two fingerprints, in [0, inf). 0 == identical signature.

    A weighted Hamming-style distance over the bucketed coordinates, plus a small graded term
    for how many size-buckets (orders of magnitude) apart the two datasets are. Modality is
    the heaviest coordinate (a text family rarely transfers to vision); shape coordinates are
    next; balance is lightest. This is a PROPOSAL-POLICY similarity, not a metric we certify
    anything with, so the weights are a transparent design choice, not a tuned hyperparameter.
    """
    if fp_a.get("key") == fp_b.get("key"):
        return 0.0
    w = {"modality": 3.0, "n_classes_bucket": 2.0, "balance_bucket": 0.5}
    d = 0.0
    for coord, weight in w.items():
        if fp_a.get(coord) != fp_b.get(coord):
            d += weight
    # graded size/feature distance: orders of magnitude apart
    for coord, weight in (("size_bucket", 1.0), ("feature_bucket", 1.0)):
        a, b = fp_a.get(coord), fp_b.get(coord)
        if a == b:
            continue
        ea, eb = _bucket_exp(a), _bucket_exp(b)
        if ea is None or eb is None:
            d += weight                      # one side unknown -> a full unit of distance
        else:
            d += weight * min(abs(ea - eb), 4) / 4.0
    return round(d, 6)


def _bucket_exp(b) -> Optional[int]:
    if not b or not str(b).startswith("1e"):
        return None
    try:
        return int(str(b)[2:])
    except ValueError:
        return None


# =========================================================================== JSONL helpers
def _append_jsonl(path: str, entry: dict) -> None:
    """Atomic append-only write (tmp + fsync + os.replace), mirroring PromotionLedger so the
    outcome log is durable and crash-safe. We rewrite the whole file under a tmp name because
    os.replace is the only portable atomic primitive here; the files are small (one line per
    move-on-a-dataset)."""
    if not isinstance(entry, dict):
        raise ValueError("entry must be a dict")
    line = json.dumps(entry, sort_keys=True)
    existing = _read_lines(path)
    payload = "\n".join([ln for ln in existing if ln.strip()] + [line]) + "\n"
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".casebase.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except OSError:
        return []


def _read_jsonl(path: str) -> List[dict]:
    out = []
    for ln in _read_lines(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


# =========================================================================== store
class CaseBaseStore:
    """Durable, warm-started proposal-policy memory.

    Two append-only JSONL files under ``vf_runs/``:
      * outcomes  -- one record per (fingerprint, family, gain, cost) realized in a run.
      * negative  -- one record per (fingerprint, family) dead end (zero gain / futility).

    Typical lifecycle (see the integration hook in the module docstring / __main__):

        store = CaseBaseStore()                         # at process start
        fp = store.compute_fingerprint(profile)         # from the loop's _profile (or records)
        priors = store.warm_start(fp)                   # per-family prior gain/cost/penalty
        # ... proposer uses `priors` to seed VoI and skip dead ends ...
        for move_name, family, gain, cost in realized:  # at run end, per executed move
            store.record_outcome(fp, family, gain, cost, move_name=move_name)

    The store NEVER reads a clock; callers may pass ``ts`` (a run-supplied monotone tag) so the
    log is reproducible. It NEVER touches the certifier.
    """

    def __init__(self, outcome_path: str = DEFAULT_OUTCOME_PATH,
                 negative_path: str = DEFAULT_NEGATIVE_PATH):
        self.outcome_path = str(outcome_path)
        self.negative_path = str(negative_path)

    # --- fingerprint passthrough (so callers need only the store) --------------------
    @staticmethod
    def compute_fingerprint(source, **kw) -> Dict:
        return fingerprint(source, **kw)

    # --- (b) append-only outcome log -------------------------------------------------
    def record_outcome(self, fp: Dict, family: str, gain: float, cost: float, *,
                       move_name: Optional[str] = None, device: Optional[str] = None,
                       ts=None) -> dict:
        """Append one realized (fingerprint, family, gain, cost) outcome. Returns the entry.

        ``device`` ("cpu" / "cuda" / ...) records WHERE the move was fit, so a GPU campaign's realized
        gains are attributable and a report can show that compute compounded -- it never affects warm-start
        ranking (purely descriptive provenance).

        Also writes a NEGATIVE-memory record when the realized gain is a dead end (<= DEAD_END_GAIN)
        so the proposer can deprioritize this (family on data-like-this) without a second call."""
        if not isinstance(fp, dict) or "key" not in fp:
            raise ValueError("fp must be a fingerprint dict (use compute_fingerprint)")
        if not isinstance(family, str) or not family:
            raise ValueError("family must be a non-empty string")
        entry = {
            "kind": "outcome",
            "fp_key": fp["key"],
            "fp": {k: fp.get(k) for k in ("modality", "size_bucket", "feature_bucket",
                                          "n_classes_bucket", "balance_bucket",
                                          "n_rows", "n_features", "n_classes")},
            "family": family,
            "move_name": move_name,
            "device": device,
            "gain": round(float(gain), 6),
            "cost": round(max(float(cost), 0.0), 6),
            "ts": ts,
        }
        _append_jsonl(self.outcome_path, entry)
        if entry["gain"] <= DEAD_END_GAIN:
            self.record_negative(fp, family, reason="zero_gain", move_name=move_name,
                                 gain=entry["gain"], ts=ts)
        return entry

    # --- (e) negative memory ---------------------------------------------------------
    def record_negative(self, fp: Dict, family_or_approach: str, *, reason: str,
                        move_name: Optional[str] = None, gain: Optional[float] = None,
                        ts=None) -> dict:
        """Record a (fingerprint, family/approach) dead end so the proposer can deprioritize it
        on similar data. ``reason`` is a free string (e.g. 'zero_gain', 'futility',
        'honest_stop_no_peek', 'diverged'). Pure memory; appends a JSONL line."""
        if not isinstance(fp, dict) or "key" not in fp:
            raise ValueError("fp must be a fingerprint dict")
        if not isinstance(family_or_approach, str) or not family_or_approach:
            raise ValueError("family_or_approach must be a non-empty string")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        entry = {
            "kind": "negative",
            "fp_key": fp["key"],
            "fp": {k: fp.get(k) for k in ("modality", "size_bucket", "feature_bucket",
                                          "n_classes_bucket", "balance_bucket")},
            "family": family_or_approach,
            "move_name": move_name,
            "reason": reason,
            "gain": None if gain is None else round(float(gain), 6),
            "ts": ts,
        }
        _append_jsonl(self.negative_path, entry)
        return entry

    # --- (d) mine an existing PromotionLedger ----------------------------------------
    def mine_promotion_ledger(self, ledger_path: str, *, profile_resolver=None) -> dict:
        """Fold an existing PromotionLedger JSONL into the outcome log: for each CERTIFIED
        promotion whose family (and, ideally, profile) can be attributed, append a positive
        outcome whose 'gain' is the certified margin above theta (observed - theta, clamped >=0)
        and whose 'cost' is unknown (1.0 placeholder, so VoI per-cost stays finite).

        The frozen promotion-ledger schema (see cross_experiment.PromotionLedger) carries
        {plan_hash, decision, metric, theta, observed, lower_bound, p_value, certified, ts} and
        does NOT always carry a family or a dataset profile. We therefore attribute a row ONLY
        when we can, via:
          * an explicit 'family' / 'fp' / 'profile' field on the row if present (newer writers
            may add it), OR
          * a caller-supplied ``profile_resolver(row) -> (family, fp_dict)`` that maps a
            plan_hash to its (family, fingerprint) using whatever side-index the caller has.
        Rows we cannot attribute are SKIPPED (counted in 'skipped'), never guessed. This keeps
        the mining honest: a beat-baseline prior only ever comes from a row we can place on a
        real (family, dataset).

        Returns a summary {'mined': int, 'skipped': int, 'total': int}.
        """
        rows = _read_jsonl(ledger_path)
        mined = skipped = 0
        for row in rows:
            if not isinstance(row, dict):
                skipped += 1
                continue
            certified = bool(row.get("certified"))
            decision = str(row.get("decision", "")).lower()
            is_win = certified or decision in ("promote", "certified")
            family = row.get("family")
            fp = row.get("fp") or row.get("fingerprint")
            if (family is None or fp is None) and profile_resolver is not None:
                try:
                    res = profile_resolver(row)
                except Exception:  # noqa: BLE001  a bad resolver must not crash mining
                    res = None
                if res:
                    fam2, fp2 = res
                    family = family or fam2
                    fp = fp or fp2
            if family is None or fp is None:
                skipped += 1
                continue
            fp_dict = fp if (isinstance(fp, dict) and "key" in fp) else _coerce_fp(fp)
            if fp_dict is None:
                skipped += 1
                continue
            theta = row.get("theta")
            observed = row.get("observed")
            if is_win and theta is not None and observed is not None:
                gain = max(0.0, float(observed) - float(theta))
            elif is_win:
                gain = 0.0
            else:
                # a non-win row is negative memory for that (family, dataset)
                self.record_negative(fp_dict, str(family), reason="ledger_not_certified",
                                     ts=row.get("ts"))
                mined += 1
                continue
            self.record_outcome(fp_dict, str(family), gain, cost=1.0,
                                move_name=row.get("move_name"), ts=row.get("ts"))
            mined += 1
        return {"mined": mined, "skipped": skipped, "total": len(rows)}

    # --- (c) warm start --------------------------------------------------------------
    def warm_start(self, fp: Dict, *, k: int = 12, max_distance: float = 6.0) -> dict:
        """Per-family PRIOR (mean gain, mean cost, count, win-rate, penalty) from the NEAREST
        past fingerprints to ``fp``.

        For every recorded outcome we compute the distance from its fingerprint to ``fp`` and
        keep the K nearest DISTINCT fingerprints within ``max_distance``. Outcomes are weighted
        by ``1 / (1 + distance)`` so identical-dataset history dominates but similar-dataset
        history still informs. We aggregate per family:
          * prior_gain   -- distance-weighted mean realized gain (clamped >= 0),
          * prior_cost   -- distance-weighted mean realized cost (>= 1e-6, so VoI stays finite),
          * n            -- raw count of contributing outcomes,
          * win_rate     -- fraction of contributing outcomes with gain > DEAD_END_GAIN,
          * penalty      -- a >= 0 deprioritization weight from NEGATIVE memory on near
                            fingerprints (more / closer dead-end records -> larger penalty).

        Returns {'fp_key', 'n_neighbors', 'families': {family: {...}}, 'order': [families,
        best-first], 'avoid': [families with penalty > 0 and no positive evidence]}.

        A new task with NO matching history yields an empty 'families' (the caller then falls
        back to the catalog priors), so this never blocks a cold start; it only helps a warm one.
        """
        outcomes = _read_jsonl(self.outcome_path)
        negatives = _read_jsonl(self.negative_path)

        # collect distinct neighbor fingerprints (by key) and their distance
        nbr_dist: Dict[str, float] = {}
        for rec in outcomes + negatives:
            key = rec.get("fp_key")
            if key is None or key in nbr_dist:
                continue
            rec_fp = _rec_fingerprint(rec)
            nbr_dist[key] = fingerprint_distance(fp, rec_fp)
        # keep K nearest within max_distance
        near = sorted(((d, key) for key, d in nbr_dist.items() if d <= max_distance))[:k]
        near_keys = {key for _, key in near}
        dist_of = {key: d for d, key in near}

        fam_gain: Dict[str, List[Tuple[float, float]]] = {}   # family -> [(weight, gain)]
        fam_cost: Dict[str, List[Tuple[float, float]]] = {}
        fam_n: Dict[str, int] = {}
        fam_wins: Dict[str, int] = {}
        for rec in outcomes:
            key = rec.get("fp_key")
            if key not in near_keys:
                continue
            fam = rec.get("family")
            if not fam:
                continue
            w = 1.0 / (1.0 + dist_of.get(key, 0.0))
            g = float(rec.get("gain", 0.0))
            c = float(rec.get("cost", 1.0))
            fam_gain.setdefault(fam, []).append((w, g))
            fam_cost.setdefault(fam, []).append((w, max(c, 1e-6)))
            fam_n[fam] = fam_n.get(fam, 0) + 1
            if g > DEAD_END_GAIN:
                fam_wins[fam] = fam_wins.get(fam, 0) + 1

        fam_penalty: Dict[str, float] = {}
        for rec in negatives:
            key = rec.get("fp_key")
            if key not in near_keys:
                continue
            fam = rec.get("family")
            if not fam:
                continue
            w = 1.0 / (1.0 + dist_of.get(key, 0.0))
            fam_penalty[fam] = fam_penalty.get(fam, 0.0) + w

        families = {}
        for fam in set(fam_gain) | set(fam_penalty):
            wg = fam_gain.get(fam, [])
            wc = fam_cost.get(fam, [])
            pg = _weighted_mean(wg)
            pc = _weighted_mean(wc) if wc else 1.0
            n = fam_n.get(fam, 0)
            families[fam] = {
                "prior_gain": round(max(0.0, pg), 6),
                "prior_cost": round(max(pc, 1e-6), 6),
                "n": n,
                "win_rate": round((fam_wins.get(fam, 0) / n), 4) if n else 0.0,
                "penalty": round(fam_penalty.get(fam, 0.0), 6),
            }

        # best-first order: positive expected-gain-per-cost first, penalty breaks ties down
        def _score(item):
            fam, st = item
            voi = st["prior_gain"] / st["prior_cost"]
            return (voi - 0.5 * st["penalty"], st["prior_gain"], -st["penalty"])

        order = [fam for fam, _ in sorted(families.items(), key=_score, reverse=True)]
        avoid = [fam for fam, st in families.items()
                 if st["penalty"] > 0.0 and st["prior_gain"] <= DEAD_END_GAIN]
        return {
            "fp_key": fp.get("key"),
            "n_neighbors": len(near_keys),
            "families": families,
            "order": order,
            "avoid": sorted(avoid),
        }

    # --- convenience: counts ---------------------------------------------------------
    def counts(self) -> dict:
        return {"outcomes": len(_read_jsonl(self.outcome_path)),
                "negatives": len(_read_jsonl(self.negative_path))}

    def summary(self, *, top: int = 10) -> dict:
        """Read-only fold over the durable memory for a campaign report: how much has accumulated, the
        strongest learned levers (per-family mean realized gain across all datasets), the dead ends, the
        distinct dataset fingerprints seen, and a compute breakdown by device (so a GPU run's contribution
        is visible). Pure reporting -- never feeds back into a run."""
        outcomes = _read_jsonl(self.outcome_path)
        negatives = _read_jsonl(self.negative_path)
        by_family: Dict[str, List[float]] = {}
        by_device: Dict[str, int] = {}
        fps = set()
        for rec in outcomes:
            fam = rec.get("family")
            if fam:
                by_family.setdefault(fam, []).append(float(rec.get("gain", 0.0)))
            by_device[rec.get("device") or "cpu"] = by_device.get(rec.get("device") or "cpu", 0) + 1
            if rec.get("fp_key"):
                fps.add(rec["fp_key"])
        mean_gain = {f: round(sum(v) / len(v), 5) for f, v in by_family.items() if v}
        top_levers = sorted(mean_gain.items(), key=lambda kv: -kv[1])[:top]
        dead_ends = sorted({rec.get("family") for rec in negatives if rec.get("family")})
        return {
            "outcome_path": self.outcome_path,
            "n_outcomes": len(outcomes),
            "n_negatives": len(negatives),
            "n_datasets": len(fps),
            "fingerprints": sorted(fps),
            "by_device": by_device,
            "top_learned_levers": top_levers,
            "dead_ends": dead_ends,
        }


def _weighted_mean(weighted: List[Tuple[float, float]]) -> float:
    if not weighted:
        return 0.0
    sw = sum(w for w, _ in weighted)
    if sw <= 0:
        return sum(v for _, v in weighted) / len(weighted)
    return sum(w * v for w, v in weighted) / sw


def _rec_fingerprint(rec: dict) -> dict:
    """Reconstruct enough of a fingerprint from a stored record to compute a distance: the
    bucket coordinates plus the key."""
    fp = dict(rec.get("fp") or {})
    fp["key"] = rec.get("fp_key")
    return fp


def _coerce_fp(fp_like) -> Optional[dict]:
    """Coerce a stored/ledger fingerprint-ish object into a fingerprint dict with a key. Accepts
    a profile dict (recompute), or a partial fp dict (derive a key from buckets if missing)."""
    if not isinstance(fp_like, dict):
        return None
    if "key" in fp_like:
        return fp_like
    # looks like a profile? recompute
    if any(k in fp_like for k in ("n_train", "n_val", "n_rows", "n_features", "kind")):
        try:
            return fingerprint(fp_like)
        except Exception:  # noqa: BLE001
            return None
    # partial bucket dict -> synthesize a key
    parts = [fp_like.get("modality", "na"), fp_like.get("size_bucket", "na"),
             fp_like.get("feature_bucket", "na"), fp_like.get("n_classes_bucket", "na"),
             fp_like.get("balance_bucket", "na")]
    out = dict(fp_like)
    out["key"] = "|".join(str(p) for p in parts)
    return out


# --------------------------------------------------------------------------- runnable example
def _example():
    """Runnable demo: warm-start is empty on a cold start, then becomes informative after a few
    simulated runs, and negative memory steers AWAY from a dead-end family. No network, no GPU."""
    import shutil

    tmpd = tempfile.mkdtemp(prefix="casebase_example_")
    try:
        store = CaseBaseStore(os.path.join(tmpd, "_casebase_outcomes.jsonl"),
                              os.path.join(tmpd, "_casebase_negative.jsonl"))

        # A small balanced tabular classification dataset (records-list path).
        records = ([{"x0": i * 0.1, "x1": -i * 0.2, "target": "a"} for i in range(600)] +
                   [{"x0": i * 0.1, "x1": i * 0.3, "target": "b"} for i in range(600)])
        fp = store.compute_fingerprint(records, kind="tabular")
        print("fingerprint:", fp["key"], "| n_rows", fp["n_rows"],
              "n_features", fp["n_features"], "balance", fp["balance_bucket"])

        cold = store.warm_start(fp)
        print("COLD warm_start -> families:", cold["families"], "| order:", cold["order"])
        assert cold["families"] == {}, "a brand-new task must have no priors"

        # Simulate three past runs on data like this: hist_gbm gains a lot, knn doesn't move,
        # logistic gains a little.
        for ts in range(3):
            store.record_outcome(fp, "hist_gbm", gain=0.12 + 0.01 * ts, cost=4.0, ts=ts)
            store.record_outcome(fp, "logistic", gain=0.03, cost=0.5, ts=ts)
            store.record_outcome(fp, "knn", gain=0.0005, cost=1.5, ts=ts)   # dead end -> negative

        warm = store.warm_start(fp)
        print("WARM warm_start order (best-first):", warm["order"])
        print("  hist_gbm:", warm["families"]["hist_gbm"])
        print("  knn     :", warm["families"]["knn"])
        print("  avoid   :", warm["avoid"])
        # hist_gbm has by far the highest realized GAIN (a strong family on data like this);
        # the order is VoI (gain-per-cost), so a cheap-and-decent family can lead, but knn (the
        # dead end) must always rank LAST and be flagged to avoid.
        best_gain = max(warm["families"], key=lambda f: warm["families"][f]["prior_gain"])
        assert best_gain == "hist_gbm", "hist_gbm must carry the highest realized prior gain"
        assert warm["order"][-1] == "knn", "the dead-end family must rank last"
        assert "knn" in warm["avoid"], "the dead-end family must be flagged to avoid"
        assert warm["families"]["hist_gbm"]["prior_gain"] > warm["families"]["knn"]["prior_gain"]
        print("counts:", store.counts())
        print("EXAMPLE OK")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == "__main__":
    _example()
