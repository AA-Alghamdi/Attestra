"""Wave-3 generative-PROPOSE bridge.

An LLM/agent AUTHORS a `build_estimator(seed)` (a novel model/pipeline, not just a catalog entry); the Wave-2
OS sandbox (`authored_pod_sandbox.run_authored`) runs it as an untrusted dedicated-uid child on a train/val
split; the prediction comes back and is scored by the FROZEN metric. The best authored candidate is then
promoted ONLY by the frozen certifier (`certify_accuracy`) on the sealed test. The certifier is never edited;
an authored model is just another candidate that must clear the same statistical bar.

Contract for authored source (enforced by the sandbox driver): it defines `build_estimator(seed)` (or `build`)
returning an object with `.fit(X, y)` and `.predict(X)`. It may import numpy/sklearn/stdlib. It runs with NO
secrets, NO network need, scrubbed env, rlimits, and reap-by-uid; only train arrays + eval features cross in,
the sealed test never does, and only a finite float `pred` vector comes back.

NOT wired into loop.py yet — this is the standalone bridge + its verification.
"""
from __future__ import annotations

import numpy as np

from .authored_pod_sandbox import run_authored


def _labels(*ys):
    s = set()
    for y in ys:
        s |= set(np.asarray(y).reshape(-1).tolist())
    return sorted(s)


def run_authored_candidate(name, source, Xtr, ytr, Xval, yval, *, metric="accuracy", labels=None,
                           cpu_s=25, mem_mb=4096, wall_s=120):
    """Run one authored candidate through the sandbox and score its val predictions with the FROZEN metric.
    Returns {name, source, ok, val_score, reason}. Never raises."""
    from vectorforge.science import score_metric
    if labels is None:
        labels = _labels(ytr, yval)
    r = run_authored(source, Xtr, ytr, Xval, cpu_s=cpu_s, mem_mb=mem_mb, wall_s=wall_s)
    if not r["ok"]:
        return {"name": name, "source": source, "ok": False, "val_score": None, "reason": r["reason"]}
    pred = np.asarray(r["pred"])
    if pred.shape[0] != np.asarray(yval).shape[0]:
        return {"name": name, "source": source, "ok": False, "val_score": None,
                "reason": f"pred len {pred.shape[0]} != yval {np.asarray(yval).shape[0]}"}
    try:
        score = float(score_metric(metric, np.asarray(yval), pred, labels))
    except Exception as e:                                   # noqa: BLE001 - report, never crash the cycle
        return {"name": name, "source": source, "ok": False, "val_score": None, "reason": f"score error: {e}"}
    return {"name": name, "source": source, "ok": True, "val_score": score, "reason": None}


def rank_authored(candidates):
    """Sort the ok candidates by val_score descending; drop the failed ones."""
    return sorted((c for c in candidates if c.get("ok")), key=lambda c: c["val_score"], reverse=True)


def certify_authored(name, source, Xtr, ytr, Xtest, ytest, *, theta, metric="accuracy", labels=None,
                     checks=1, alpha=0.05, cpu_s=25, mem_mb=4096, wall_s=120):
    """Promote-or-not: run the authored model on the SEALED test through the sandbox, score with the frozen
    metric, and let the FROZEN certifier decide. Returns the certifier dict (+name) or a not-certified reason."""
    from vectorforge.science import score_metric, certify_accuracy
    if labels is None:
        labels = _labels(ytr, ytest)
    r = run_authored(source, Xtr, ytr, Xtest, cpu_s=cpu_s, mem_mb=mem_mb, wall_s=wall_s)
    if not r["ok"]:
        return {"name": name, "certified": False, "reason": f"sandbox: {r['reason']}"}
    pred = np.asarray(r["pred"])
    if pred.shape[0] != np.asarray(ytest).shape[0]:
        return {"name": name, "certified": False, "reason": "pred/test length mismatch"}
    observed = float(score_metric(metric, np.asarray(ytest), pred, labels))
    cert = certify_accuracy(observed, int(np.asarray(ytest).shape[0]), theta, checks=checks, alpha=alpha)
    cert["name"] = name
    return cert


# --------------------------------------------------------------------------- stub authors (no-API testing)
# Deterministic stand-ins for the LLM author: each is a self-contained `build_estimator(seed)` source defining
# a NON-trivial pipeline (not a bare catalog ctor) so the bridge is exercised exactly as a real authored model
# would be. The real LLM/agent path returns sources in the same shape.
def stub_authored_sources():
    H = "import numpy as np\n"
    return [
        ("pca_rbf_svc", H +
         "from sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "from sklearn.decomposition import PCA\n"
         "from sklearn.svm import SVC\n"
         "def build_estimator(seed):\n"
         "    return make_pipeline(StandardScaler(), PCA(n_components=30, random_state=seed),\n"
         "                         SVC(C=4.0, gamma='scale', random_state=seed))\n"),
        ("scaled_hist_gb", H +
         "from sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "from sklearn.ensemble import HistGradientBoostingClassifier\n"
         "def build_estimator(seed):\n"
         "    return make_pipeline(StandardScaler(),\n"
         "                         HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,\n"
         "                                                        max_depth=None, random_state=seed))\n"),
        ("bagged_extra_trees", H +
         "from sklearn.ensemble import ExtraTreesClassifier\n"
         "def build_estimator(seed):\n"
         "    return ExtraTreesClassifier(n_estimators=600, max_features='sqrt',\n"
         "                                bootstrap=True, random_state=seed, n_jobs=1)\n"),
        ("interaction_logreg", H +
         "from sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "from sklearn.kernel_approximation import Nystroem\n"
         "from sklearn.linear_model import LogisticRegression\n"
         "def build_estimator(seed):\n"
         "    return make_pipeline(StandardScaler(),\n"
         "                         Nystroem(n_components=200, random_state=seed),\n"
         "                         LogisticRegression(C=2.0, max_iter=400))\n"),
    ]
