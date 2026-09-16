"""Pre-registration commitment layer (mechanizes no-anchoring / no-theta-relaxation).

The scientific-integrity rule of this project is: you may not relax an STL/metric spec
threshold to make a result "pass" without that change being VISIBLE. This module
mechanizes that rule. A spec is normalized into a canonical, JSON-serializable form and
content-addressed via ``vectorforge.science.digest``. Two semantically-identical specs
produce the same ``plan_hash``; ANY field change -- crucially including a *relaxed
threshold* -- produces a DIFFERENT ``plan_hash``.

A relaxed threshold is therefore a NEW commitment with a NEW hash, recorded in the
append-only ``PlanRegistry`` ALONGSIDE the old one. The original commitment is never
overwritten or deleted, so post-hoc spec-bending is always auditable: the registry shows
both the pre-registered threshold and the relaxed one, with distinct hashes and order of
arrival. There is no code path that silently mutates a prior commitment.

This module is self-contained: it imports only the stdlib and (read-only)
``vectorforge.science.digest`` for hash reuse. It is NOT wired into the loop; the operator
hand-wires it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Read-only import: we reuse science.digest for the content address. We do NOT modify
# science.py in any way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorforge.science import digest  # noqa: E402

# Fixed rounding rule for float normalization. Any spec field that is a float is rounded to
# this many decimal places before hashing, so that 0.8 and 0.8000000001 (e.g. from a YAML
# round-trip) canonicalize identically, while a genuine threshold change (0.80 -> 0.78)
# still produces a distinct canonical value and hence a distinct hash.
_FLOAT_DECIMALS = 9


def _round_floats(obj):
    """Recursively round floats with the fixed rule; leave everything else structurally intact.

    Bools are NOT floats and are left alone (isinstance(True, int) is True but not float).
    NaN/Inf are passed through as their repr is stable and JSON-serialization is handled by
    the caller (we never expect them in a spec, but we do not silently corrupt them).
    """
    if isinstance(obj, float):
        return round(obj, _FLOAT_DECIMALS)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def _canonicalize(obj):
    """Return a structure whose repr is stable: dict keys sorted, tuples -> lists, floats rounded.

    science.digest hashes ``repr(obj)``. Python dict repr is insertion-ordered, so to make
    the hash invariant to key insertion order we recursively rebuild every dict with sorted
    keys. Tuples are normalized to lists so a tuple and an equivalent list hash identically
    (JSON has no tuple type anyway). Floats are rounded with the fixed rule.
    """
    obj = _round_floats(obj)
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def canonical_spec(*, metric, threshold, alpha, forbidden_fields=(),
                   selection_rule="best_val_lower_bound", extra=None):
    """Build a normalized, sorted, JSON-serializable pre-registration spec.

    Parameters
    ----------
    metric : str
        Name of the certified metric (e.g. "balanced_accuracy", "neg_rmse").
    threshold : float
        The pre-registered theta the metric's lower bound must clear. Relaxing this value
        is the canonical "spec-bend" the registry is designed to make visible.
    alpha : float
        The pre-registered significance level for the one-sided certificate.
    forbidden_fields : iterable of str
        Fields the experiment is forbidden to tune/anchor on (e.g. test-set labels, the
        target paper's reported numbers). Sorted and de-duplicated for canonical order.
    selection_rule : str
        The pre-committed selection rule (default the project's select-then-bound rule).
    extra : dict or None
        Optional additional pre-registered fields (any JSON-serializable mapping). Merged
        under the "extra" key; recursively canonicalized.

    Returns
    -------
    dict : canonical spec with a stable key order. JSON-serializable.
    """
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")
    if not isinstance(selection_rule, str) or not selection_rule:
        raise ValueError("selection_rule must be a non-empty string")
    try:
        threshold = float(threshold)
        alpha = float(alpha)
    except (TypeError, ValueError) as e:
        raise ValueError(f"threshold and alpha must be floats: {e}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1): {alpha}")

    # sorted + de-duplicated forbidden fields (stable canonical order regardless of input order)
    ff = sorted({str(x) for x in forbidden_fields})

    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError("extra must be a dict or None")

    # Build with sorted keys at the top level too, then canonicalize recursively. The
    # canonical version (sorted keys, rounded floats, tuples->lists) is what gets hashed and
    # is what we return, so callers always see exactly what was committed.
    spec = {
        "alpha": alpha,
        "extra": extra,
        "forbidden_fields": ff,
        "metric": metric,
        "schema": "vfplatform.prereg/v1",
        "selection_rule": selection_rule,
        "threshold": threshold,
    }
    canon = _canonicalize(spec)
    # Sanity: must be JSON-serializable (fail loud at commit time, not silently at audit time).
    json.dumps(canon, sort_keys=True)
    return canon


def commit(spec: dict) -> dict:
    """Content-address a canonical spec.

    Re-canonicalizes the input (so a caller passing a hand-built dict still gets a stable
    hash) and returns ``{"plan_hash": <science.digest>, "spec": <canonical spec>}``.

    Same spec -> same hash. Any field change (including a relaxed threshold) -> a different
    hash. The hash is computed via ``vectorforge.science.digest`` over the canonical spec, so
    it is identical to the addressing scheme the rest of the platform uses.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict")
    canon = _canonicalize(spec)
    return {"plan_hash": digest(canon), "spec": canon}


class PlanRegistry:
    """Append-only JSONL ledger of pre-registration commitments.

    Every ``register`` writes one JSON line and is durable (tmp file + fsync + os.replace of
    the whole file). The registry NEVER mutates or removes a prior line: a relaxed-threshold
    spec is appended as a new commitment with a new hash next to the original, so any
    spec-bend is visible in the file's history. Tolerant of a missing or partially-corrupt
    file (corrupt lines are skipped on read, not silently dropped from disk).
    """

    def __init__(self, path):
        self.path = str(path)

    def _read_lines(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read().splitlines()
        except OSError:
            return []

    def register(self, commitment: dict) -> None:
        """Atomically append one commitment. Read-modify-(atomic)write of the whole file.

        We rewrite the whole file via tmp+fsync+os.replace so the file is never observed in a
        half-written state (append-mode partial writes can corrupt the last line on crash).
        Existing valid lines are preserved verbatim; the new line is added at the end.
        """
        if not isinstance(commitment, dict):
            raise ValueError("commitment must be a dict")
        line = json.dumps(commitment, sort_keys=True)
        existing = self._read_lines()
        existing = [ln for ln in existing if ln.strip()]
        payload = ("\n".join(existing + [line]) + "\n")

        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".prereg.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def all(self) -> list:
        """Return every well-formed commitment as a list of dicts (corrupt lines skipped)."""
        out = []
        for ln in self._read_lines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except (json.JSONDecodeError, ValueError):
                # tolerant: skip a corrupt line rather than crashing the audit
                continue
        return out

    def count(self) -> int:
        return len(self.all())


# --------------------------------------------------------------------------- self-test
def _selftest():
    import shutil

    tmpd = tempfile.mkdtemp(prefix="prereg_selftest_")
    try:
        s1 = canonical_spec(metric="balanced_accuracy", threshold=0.80, alpha=0.05,
                            forbidden_fields=("test_labels", "paper_number"))
        s1b = canonical_spec(metric="balanced_accuracy", threshold=0.80, alpha=0.05,
                             forbidden_fields=("paper_number", "test_labels"))  # reordered
        assert commit(s1)["plan_hash"] == commit(s1b)["plan_hash"], "order must not matter"
        s2 = canonical_spec(metric="balanced_accuracy", threshold=0.78, alpha=0.05,
                            forbidden_fields=("test_labels", "paper_number"))  # relaxed
        assert commit(s1)["plan_hash"] != commit(s2)["plan_hash"], "relaxed threshold must differ"

        reg = PlanRegistry(os.path.join(tmpd, "plans.jsonl"))
        assert reg.count() == 0
        reg.register(commit(s1))
        reg.register(commit(s2))
        assert reg.count() == 2
        hashes = {c["plan_hash"] for c in reg.all()}
        assert len(hashes) == 2
        print("prereg self-test OK")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
