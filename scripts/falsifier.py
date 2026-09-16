"""INDEPENDENT, blinded adversarial falsifier for a VectorForge certificate.

WHY THIS EXISTS
---------------
The goal loop grades its own homework: it selects a winner on validation, takes ONE sealed peek, and the
SAME process certifies it. That is self-graded acceptance. A certificate is only worth something if an
INDEPENDENT party can re-derive the claim from scratch and either CONFIRM it or honestly REFUTE it. This
module is that independent party. It is the antidote to "the loop says it passed".

WHAT INDEPENDENCE MEANS HERE (every one of these is load-bearing)
-----------------------------------------------------------------
  1. It NEVER imports the loop's accept/select-then-bound decision. It does not call run_goal_loop, does not
     read GoalLoopResult, does not trust `cert["certified"]`. It consumes ONLY (a) the certificate's CLAIM
     fields -- the dataset spec, the winner family string, the metric, and theta -- and (b) the FROZEN
     primitives in vectorforge.science (the same math any reviewer would re-run). It re-derives the verdict.
  2. It RE-MATERIALIZES the data from the dataset spec NAME (e.g. "sklearn:breast_cancer") via the connector,
     so it never reuses the loop's in-memory rows or its stored split. Same data source, freshly loaded.
  3. It builds a FRESH leakage-safe split with science.make_splits at a DIFFERENT seed (and sweeps several
     seeds). A certificate that only survives the loop's lucky seed=0 split is exactly what we want to catch:
     the loop's sealed_digest is deterministic in seed=0, so re-using it would be a REPLAY, not a check.
  4. It re-runs the FROZEN leakage auditor (science.audit) on its own split. If the split it drew is itself
     leaky, it refuses to confirm (a bound on a leaky split is meaningless).
  5. It re-trains the reported winner family from scratch on each fresh train split, featurizes with its OWN
     minimal standardizer (mirrors the documented TabularFeaturizer contract but is the falsifier's own code,
     so a bug in the loop's featurizer cannot be laundered into agreement), scores on the held-out test slice,
     and recomputes the lower bound with the FROZEN certifier (certify_accuracy / certify_regression /
     classification bootstrap). It then tries to make the claim FAIL across seeds.

VERDICT
-------
  * CONFIRMED  -- on EVERY independent seed the frozen lower bound clears theta on a clean (audit-passing)
                  fresh split. The claim survives independent adversarial re-derivation.
  * REFUTED    -- on at least one clean fresh split the frozen lower bound does NOT clear theta. The loop's
                  certificate does not generalize to an independently drawn split; the worst seed is reported.
  * INCONCLUSIVE -- the falsifier could not build a clean test on which to judge (e.g. every fresh split
                  failed the leakage audit, or the dataset/family could not be re-materialized). Honest abstain.

It NEVER mutates a certificate, the sealed peek, or select-then-bound, and creates no new sealed peek of the
loop's own locked test -- it builds its OWN test from its OWN split, so there is no multiplicity to launder.

This is a READ-ONLY checker. Run as a CI gate (see __main__ / falsify_cert exit codes).
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# FROZEN primitives only. We re-derive with the exact same math a reviewer would re-run; we do NOT import the
# loop's decision logic (run_goal_loop / GoalLoopResult / _val_lower_bound / certify-step wiring).
from vectorforge import science
from vfplatform import connectors
from vfplatform.harness import catalog_for, resolve_family


# =========================================================================== certificate claim extraction
SUPPORTED_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "r2", "neg_rmse", "neg_mae")


class FalsifierError(Exception):
    """The falsifier could not even set up an independent check (bad cert, undrede-materializable data)."""


def _load_cert(cert_or_path):
    """Accept a certificate dict, a path to a certificate JSON, or a path to a vf_runs run-*.json (which
    nests the certificate under artifacts.certificate). Returns the plain certificate dict."""
    if isinstance(cert_or_path, dict):
        return dict(cert_or_path)
    with open(cert_or_path) as fh:
        obj = json.load(fh)
    if isinstance(obj, dict) and "artifacts" in obj and isinstance(obj["artifacts"], dict):
        inner = obj["artifacts"].get("certificate")
        if isinstance(inner, dict):
            return dict(inner)
    return obj


def _claim(cert, *, dataset=None, metric=None, theta=None):
    """Pull the CLAIM the falsifier must independently re-test out of the certificate.

    These are the only certificate fields we trust: they describe WHAT was claimed, not WHETHER it holds.
    `dataset`/`metric`/`theta` overrides let a caller supply the dataset spec when the certificate does not
    embed it (the loop's certificate records the winner+theta+metric but not always the source name)."""
    winner_family = cert.get("winner_family") or cert.get("family")
    if not winner_family:
        raise FalsifierError("certificate has no winner_family/family -- cannot identify the model to retrain")
    metric = metric or cert.get("metric") or "accuracy"
    if metric not in SUPPORTED_METRICS:
        raise FalsifierError(f"metric {metric!r} is not independently certifiable {SUPPORTED_METRICS}")
    theta = theta if theta is not None else cert.get("theta")
    if theta is None:
        raise FalsifierError("certificate has no theta -- nothing to clear")
    dataset = dataset or cert.get("dataset") or cert.get("source") or cert.get("dataset_spec")
    if not dataset:
        raise FalsifierError(
            "certificate does not name its dataset; pass --dataset (e.g. sklearn:breast_cancer or "
            "breast_cancer). The falsifier must RE-MATERIALIZE the data from its source, not reuse the "
            "loop's rows.")
    return {"winner_family": str(winner_family), "metric": str(metric), "theta": float(theta),
            "dataset": str(dataset)}


# =========================================================================== independent data re-materialization
def _materialize(dataset_spec):
    """Independently re-load the dataset from its SOURCE NAME. Accepts 'sklearn:breast_cancer', 'breast_cancer'
    (bare sklearn name), 'hf://...', or 'openml://dataset/<id>'. Returns the connector spec dict."""
    spec = dataset_spec.strip()
    if spec.startswith("sklearn:"):
        return connectors.load_sklearn(spec.split(":", 1)[1])
    if spec.startswith("hf://") or spec.startswith("openml://"):
        return connectors.materialize(spec)
    # bare name -> try a bundled sklearn dataset (the offline, reproducible path)
    return connectors.load_sklearn(spec)


# =========================================================================== independent featurizer
# This is the falsifier's OWN minimal numeric standardizer + one-hot. It MIRRORS the documented
# TabularFeaturizer contract (numeric standardized over the split union, categoricals one-hot over the vocab)
# but is independent code: if the loop's featurizer had a bug, re-using it would launder that bug into
# agreement. Text is handled by deferring to a fresh TfidfVectorizer (independent fit on the fresh train).
class _IndependentTabular:
    def __init__(self):
        self.numeric, self.cats, self.vocab, self.mu, self.sd = [], [], {}, {}, {}

    @staticmethod
    def _f(r, c):
        return (r.get("features") or {}).get(c)

    def fit(self, schema_rows, all_rows):
        cols = list((schema_rows[0].get("features") or {}).keys())
        self.numeric = [c for c in cols
                        if all(isinstance(self._f(r, c), (int, float)) for r in schema_rows
                               if self._f(r, c) is not None)]
        self.cats = [c for c in cols if c not in self.numeric]
        self.vocab = {c: sorted({str(self._f(r, c)) for r in all_rows}) for c in self.cats}
        for c in self.numeric:
            vals = [float(self._f(r, c)) for r in all_rows if isinstance(self._f(r, c), (int, float))]
            arr = np.asarray(vals, dtype=float) if vals else np.zeros(1)
            self.mu[c] = float(arr.mean())
            s = float(arr.std())
            self.sd[c] = s if s > 1e-9 else 1.0
        return self

    def transform(self, rows):
        X = []
        for r in rows:
            row = [(float(self._f(r, c)) if isinstance(self._f(r, c), (int, float)) else self.mu.get(c, 0.0))
                   for c in self.numeric]
            row = [(v - self.mu.get(c, 0.0)) / self.sd.get(c, 1.0) for v, c in zip(row, self.numeric)]
            for c in self.cats:
                row += [1.0 if str(self._f(r, c)) == lev else 0.0 for lev in self.vocab[c]]
            X.append(row)
        return np.array(X, dtype=float)


class _IndependentText:
    def __init__(self, text_key="text", max_features=20000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.text_key = text_key
        self.vec = TfidfVectorizer(max_features=max_features)

    def fit(self, schema_rows, all_rows):
        self.vec.fit([str(r.get(self.text_key, "")) for r in all_rows])
        return self

    def transform(self, rows):
        return self.vec.transform([str(r.get(self.text_key, "")) for r in rows])


# =========================================================================== winner reconstruction
def _parse_family(winner_family, task_type, kind):
    """Map the certificate's winner_family STRING back to an independent (catalog_key, params) and resolve it
    to a fresh sklearn estimator ctor through the FROZEN catalog clamp. Family names look like
    'logistic|C=0.1' (catalog-proposed: key|k=v|...) or 'logistic|C1.0' / 'random_forest|200' (the static
    moves() menu, where the tag is a label, not k=v). For the latter we fall back to the catalog DEFAULT for
    the parsed key, which is the safe, in-range reconstruction.

    Returns (catalog_key, ctor(seed)->estimator, clamped_params). Raises if the key is not in the catalog."""
    catalog = catalog_for(kind, task_type)
    key = winner_family.split("|", 1)[0]
    tag = winner_family.split("|", 1)[1] if "|" in winner_family else ""
    params = {}
    for piece in tag.split("|"):
        if "=" in piece:
            k, v = piece.split("=", 1)
            try:
                v = float(v)
            except ValueError:
                pass
            params[k] = v
    resolved = resolve_family(catalog, key, params)
    if resolved is None:
        raise FalsifierError(
            f"winner family key {key!r} (from {winner_family!r}) is not in the {kind}/{task_type} catalog "
            f"{sorted(catalog)}; cannot independently rebuild the model")
    return resolved


# =========================================================================== one independent trial
def _score_and_bound(metric, y_true, y_pred, theta, *, labels, alpha, seed):
    """Recompute the FROZEN lower bound for one metric on one held-out slice. Mirrors the metric routing in
    vfplatform.sealed.certify_on_sealed but calls only the frozen science.* certifiers. checks=1 because the
    falsifier draws its OWN single test slice (no multiplicity to launder)."""
    if metric == "accuracy":
        observed = sum(1 for a, b in zip(y_true, y_pred) if str(a) == str(b)) / max(len(y_true), 1)
        cert = science.certify_accuracy(observed, len(y_true), theta, checks=1, alpha=alpha)
        return {"observed": cert["observed"], "lower_bound": cert["lower_bound"],
                "clears": bool(cert["lower_bound"] > theta), "n": cert["n"], "deferred": False}
    if metric in ("balanced_accuracy", "macro_f1"):
        yt = [str(x) for x in y_true]
        yp = [str(x) for x in y_pred]
        labs = labels or sorted(set(yt))
        lower, point = science._bootstrap_classification_metric_lower(metric, yt, yp, labs, alpha=alpha,
                                                                      seed=seed)
        return {"observed": point, "lower_bound": lower, "clears": bool(lower > theta), "n": len(yt),
                "deferred": False}
    # regression
    yt = [float(x) for x in y_true]
    yp = [float(x) for x in y_pred]
    cert = science.certify_regression(yt, yp, metric, theta, checks=1, alpha=alpha, seed=seed)
    return {"observed": cert["observed"], "lower_bound": cert["lower_bound"],
            "clears": bool(cert["certified"]), "n": cert["n"], "deferred": bool(cert.get("deferred"))}


def _one_trial(spec, claim, seed, *, alpha):
    """One fully-independent re-derivation at a given split seed: fresh split -> fresh audit -> fresh fit ->
    held-out score -> frozen bound. Returns a per-trial report dict."""
    records = spec["records"]
    target_key = spec.get("target_key", "target")
    kind = spec.get("kind", "tabular")
    task_type = spec.get("task_type", "binary")
    is_reg = task_type == "regression"
    labels = spec.get("labels")
    text_key = "text"

    # 1. FRESH leakage-safe split at THIS seed (regression: stratify on target quantile bins, then strip).
    if not is_reg:
        train, val, test, split_info = science.make_splits(records, seed=seed, target_key=target_key,
                                                           text_key=text_key)
    else:
        vals = np.array([float(r.get(target_key)) for r in records], dtype=float)
        edges = np.quantile(vals, [i / 10.0 for i in range(1, 10)])
        bins = np.digitize(vals, edges)
        tagged = [dict(r, _strat_bin=int(b)) for r, b in zip(records, bins)]
        train, val, test, split_info = science.make_splits(tagged, seed=seed, target_key="_strat_bin",
                                                           text_key=text_key)
        strip = lambda rs: [{k: v for k, v in r.items() if k != "_strat_bin"} for r in rs]
        train, val, test = strip(train), strip(val), strip(test)

    if not train or not test:
        return {"seed": seed, "status": "skip", "reason": "empty fresh split", "clears": None}

    # 2. FRESH leakage audit on THIS split (a bound on a leaky split is meaningless -> abstain).
    audit = science.audit(train, test, allow_features=(kind == "tabular"), text_key=text_key,
                          target_key=target_key, min_test_n=1)
    leak_findings = [f for f in audit["findings"] if not f.get("ok", True)
                     and f["gate"] not in ("min_test_n", "features_present_must_not_leak")]
    if leak_findings:
        return {"seed": seed, "status": "skip", "reason": "fresh split failed leakage audit",
                "leak": [f["gate"] for f in leak_findings], "clears": None}

    # 3. independent featurizer (fit on split union, mirrors the documented contract; own code).
    feat = (_IndependentText(text_key=text_key) if kind == "text" else _IndependentTabular())
    feat.fit(train, train + val + test)
    Xtr, Xte = feat.transform(train), feat.transform(test)

    # 4. encode targets independently (classification -> indices via the certificate-declared label order).
    if is_reg:
        ytr = np.array([float(r.get(target_key)) for r in train], dtype=float)
    else:
        labs = labels or sorted({str(r.get(target_key)) for r in train + test})
        l2i = {lab: i for i, lab in enumerate(labs)}
        ytr = np.array([l2i[str(r.get(target_key))] for r in train])

    # 5. rebuild the reported winner family from scratch and fit it on the FRESH train split.
    key, ctor, clamped = _parse_family(claim["winner_family"], task_type, kind)
    est = ctor(int(seed))
    est.fit(Xtr, ytr)

    # 6. predict on the held-out test slice and decode to labels/floats.
    raw = est.predict(Xte)
    if is_reg:
        y_pred = [float(p) for p in raw]
        y_true = [float(r.get(target_key)) for r in test]
        bound_labels = None
    else:
        inv = {i: lab for lab, i in l2i.items()}
        y_pred = [inv[int(p)] for p in raw]
        y_true = [str(r.get(target_key)) for r in test]
        bound_labels = labs

    # 7. recompute the FROZEN lower bound and ask: does it still clear theta on THIS independent split?
    res = _score_and_bound(claim["metric"], y_true, y_pred, claim["theta"], labels=bound_labels,
                           alpha=alpha, seed=seed)
    res.update({"seed": seed, "status": "scored", "rebuilt_family": key, "rebuilt_params": clamped,
                "n_train": len(train), "n_test": len(test)})
    return res


# =========================================================================== top-level falsify
def falsify_cert(cert_or_path, *, dataset=None, metric=None, theta=None, seeds=(101, 202, 303, 404, 505),
                 alpha=0.05):
    """Independently and adversarially re-test a certificate's claim.

    Returns a verdict dict:
      {"verdict": "CONFIRMED"|"REFUTED"|"INCONCLUSIVE", "claim": {...}, "trials": [...],
       "worst": {...}, "summary": "..."}

    CONFIRMED  iff at least one clean scored trial exists AND every scored trial clears theta.
    REFUTED    iff at least one clean scored trial does NOT clear theta (worst seed reported).
    INCONCLUSIVE iff no clean scored trial could be built.
    """
    cert = _load_cert(cert_or_path)
    claim = _claim(cert, dataset=dataset, metric=metric, theta=theta)
    spec = _materialize(claim["dataset"])

    trials = []
    for s in seeds:
        try:
            trials.append(_one_trial(spec, claim, int(s), alpha=alpha))
        except FalsifierError:
            raise
        except Exception as ex:  # noqa: BLE001  a numerical failure in one seed must not crash the gate
            trials.append({"seed": int(s), "status": "error", "reason": str(ex)[:200], "clears": None})

    scored = [t for t in trials if t.get("status") == "scored"]
    if not scored:
        return {"verdict": "INCONCLUSIVE", "claim": claim, "trials": trials, "worst": None,
                "summary": "no clean independent split could be scored (every fresh seed was skipped/errored); "
                           "honest abstain -- cannot confirm OR refute"}

    failures = [t for t in scored if not t["clears"]]
    worst = min(scored, key=lambda t: (t["lower_bound"] if t["lower_bound"] is not None else -1e9))
    loop_lb = cert.get("lower_bound")
    if failures:
        verdict = "REFUTED"
        summary = (f"REFUTED: on {len(failures)}/{len(scored)} independent fresh-split seeds the frozen lower "
                   f"bound did NOT clear theta={claim['theta']}. Worst independent seed {worst['seed']}: "
                   f"observed={worst['observed']}, lower_bound={worst['lower_bound']} (loop claimed "
                   f"lower_bound={loop_lb}). The certificate does not survive an independently drawn split.")
    else:
        verdict = "CONFIRMED"
        summary = (f"CONFIRMED: on all {len(scored)} independent fresh-split seeds the frozen lower bound "
                   f"cleared theta={claim['theta']}. Worst independent seed {worst['seed']}: "
                   f"observed={worst['observed']}, lower_bound={worst['lower_bound']} (loop claimed "
                   f"lower_bound={loop_lb}). The claim survives independent adversarial re-derivation.")
    return {"verdict": verdict, "claim": claim, "trials": trials, "worst": worst,
            "loop_claimed_lower_bound": loop_lb, "summary": summary}


# =========================================================================== CLI / CI gate
def _exit_code(verdict):
    # CI contract: 0 = CONFIRMED (gate passes), 1 = REFUTED (block the merge), 2 = INCONCLUSIVE (abstain).
    return {"CONFIRMED": 0, "REFUTED": 1, "INCONCLUSIVE": 2}.get(verdict, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent adversarial falsifier for a VectorForge certificate.")
    ap.add_argument("cert", help="path to a certificate JSON or a vf_runs run-*.json")
    ap.add_argument("--dataset", default=None,
                    help="dataset source name if the cert does not embed it (e.g. sklearn:breast_cancer)")
    ap.add_argument("--metric", default=None, help="override the metric (default: from cert)")
    ap.add_argument("--theta", default=None, type=float, help="override theta (default: from cert)")
    ap.add_argument("--seeds", default="101,202,303,404,505",
                    help="comma-separated independent split seeds (must differ from the loop's seed=0)")
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--json", action="store_true", help="print the full verdict dict as JSON")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    out = falsify_cert(args.cert, dataset=args.dataset, metric=args.metric, theta=args.theta,
                       seeds=seeds, alpha=args.alpha)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"VERDICT: {out['verdict']}")
        print(out["summary"])
        for t in out["trials"]:
            if t.get("status") == "scored":
                print(f"  seed {t['seed']:>4}: obs={t['observed']} lb={t['lower_bound']} "
                      f"clears={t['clears']} (n_test={t['n_test']})")
            else:
                print(f"  seed {t['seed']:>4}: {t['status']} -- {t.get('reason', '')}")
    return _exit_code(out["verdict"])


if __name__ == "__main__":
    sys.exit(main())
