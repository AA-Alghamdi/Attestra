"""RunPod serverless worker handler for VectorForge GPU fan-out.

Runs INSIDE the worker container (which ships a matched torch + numpy + CUDA stack). It does NOT promote:
it fits a candidate and returns either the VALIDATION score (for the leaderboard) or PREDICTIONS on the
sealed test (op='fit_predict_test') so the FROZEN certifier can run on the LOCAL machine. The worker never
runs the certifier and never decides certification -- it is pure compute.

Device-agnostic: uses torch (GPU when available) for torch_mlp_* families, sklearn otherwise. The sklearn
path lets the handler be smoke-tested locally with no torch/GPU.

Input (job["input"]):
  {op: "fit_val" | "fit_predict_test", task_type, family, params, seed, metric,
   Xtr, ytr, Xva|Xte, yva (fit_val only)}   -- arrays as nested lists
Output:
  fit_val        -> {val_score, latency_ms, device, family}
  fit_predict_test -> {y_pred, latency_ms, device, family}
"""
import time

import numpy as np


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _cuda_diag():
    """Explain WHY a run is on cpu vs cuda. A silent cpu fallback on a GPU host (e.g. a torch wheel whose
    bundled CUDA runtime is newer than the host driver) would let a 'GPU' certificate be produced on CPU;
    this surfaces the cause so that can never pass unnoticed."""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        try:
            info["device_count"] = int(torch.cuda.device_count())
        except Exception as ex:  # noqa: BLE001
            info["device_count_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
        if not info["cuda_available"]:
            try:
                torch.cuda.init()
            except Exception as ex:  # noqa: BLE001
                info["cuda_error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
    except Exception as ex:  # noqa: BLE001
        info["import_error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
    return info


def build_model(family, params, seed, task_type, n_classes):
    """Reconstruct an estimator from a serializable spec (no lambdas over the wire)."""
    params = params or {}
    if family.startswith("torch_mlp"):
        from torch_models import TorchMLPClassifier, TorchMLPRegressor  # in the image
        hidden = tuple(int(h) for h in str(params.get("hidden", "64")).split("x"))
        kw = dict(hidden=hidden, dropout=float(params.get("dropout", 0.1)),
                  lr=float(params.get("lr", 1e-3)), weight_decay=float(params.get("weight_decay", 0.0)),
                  epochs=int(params.get("epochs", 60)), seed=seed)
        return TorchMLPRegressor(**kw) if task_type == "regression" else \
            TorchMLPClassifier(n_classes=n_classes, **kw)
    if family.startswith("torch_cnn"):
        from torch_models import TorchCNNClassifier   # GPU spatial arm (2D conv on square inputs, else 1D)
        return TorchCNNClassifier(n_classes=n_classes, ch=int(params.get("ch", 16)),
                                  lr=float(params.get("lr", 1e-3)), epochs=int(params.get("epochs", 40)),
                                  weight_decay=float(params.get("weight_decay", 0.0)), seed=seed)
    # sklearn fallbacks (also the local smoke-test path). Kept in sync with vfplatform/harness.py move menu.
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                                  RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesClassifier)
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    table = {
        "logistic": lambda: LogisticRegression(max_iter=2000, C=float(params.get("C", 1.0))),
        "ridge": lambda: Ridge(alpha=float(params.get("alpha", 1.0))),
        "random_forest": lambda: RandomForestClassifier(n_estimators=int(params.get("n", 200)),
                                                        max_depth=params.get("max_depth"), random_state=seed),
        "hist_gbm": lambda: HistGradientBoostingClassifier(max_iter=int(params.get("it", 200)),
                                                          max_depth=params.get("max_depth"), random_state=seed),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=int(params.get("n", 300)), random_state=seed),
        "svc_rbf": lambda: SVC(C=float(params.get("C", 1.0)), probability=True, random_state=seed),
        "calibrated_gbm": lambda: CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=150, random_state=seed), method="isotonic", cv=3),
        "random_forest_reg": lambda: RandomForestRegressor(n_estimators=int(params.get("n", 200)), random_state=seed),
        "hist_gbm_reg": lambda: HistGradientBoostingRegressor(max_iter=int(params.get("it", 200)), random_state=seed),
    }
    key = family.split("|")[0]
    if key not in table:
        raise ValueError(f"unknown family {family!r}")
    return table[key]()


def _score(task_type, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task_type == "regression":
        ss_res = float(((y_true - y_pred) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) or 1e-9
        return 1.0 - ss_res / ss_tot
    return float((y_pred == y_true).mean())


def _confidences(task_type, model, X):
    """Per-sample max class probability for a classifier (for the platform's ECE), or None when the model
    exposes no calibrated probability. Never raises -- a missing channel just disables ECE downstream."""
    if task_type == "regression":
        return None
    try:
        proba = model.predict_proba(np.asarray(X, dtype=float))
        return np.max(np.asarray(proba, dtype=float), axis=1).tolist()
    except Exception:  # noqa: BLE001  (LinearSVC etc. have no predict_proba)
        return None


def run(inp):
    op = inp["op"]
    task_type = inp.get("task_type", "binary")
    Xtr = np.asarray(inp["Xtr"], dtype=float)
    ytr = np.asarray(inp["ytr"])
    n_classes = int(inp.get("n_classes", len(set(ytr.tolist())))) if task_type != "regression" else 1
    model = build_model(inp["family"], inp.get("params"), int(inp.get("seed", 0)), task_type, n_classes)
    t0 = time.time()
    model.fit(Xtr, ytr)
    if op == "fit_val":
        # CONTRACT (step 7/F6): return val PREDICTIONS (+ confidences), NOT a worker-computed scalar. The
        # platform re-scores locally with the frozen score_metric, so a buggy/malicious worker cannot steer
        # selection via a faked val_score, the metric used is the real one (not the worker's accuracy-only
        # _score), and the val labels (yva) never need to be shipped to the worker.
        Xva = np.asarray(inp["Xva"], dtype=float)
        pred = model.predict(Xva)
        return {"y_pred_val": np.asarray(pred).tolist(),
                "proba_val": _confidences(task_type, model, Xva),    # for platform ECE (step 8); may be None
                "latency_ms": (time.time() - t0) * 1000.0, "device": _device(),
                "cuda": _cuda_diag(), "family": inp["family"]}
    if op == "fit_predict_test":
        pred = model.predict(np.asarray(inp["Xte"], dtype=float))
        # echo the platform-supplied per-row ids ALIGNED to predictions so the platform can reindex/assert
        # 1:1 coverage before scoring the sealed test (step 7/F5: catches truncation/reorder/substitution).
        return {"y_pred": np.asarray(pred).tolist(), "row_ids": inp.get("row_ids"),
                "latency_ms": (time.time() - t0) * 1000.0,
                "device": _device(), "cuda": _cuda_diag(), "family": inp["family"]}
    raise ValueError(f"unknown op {op!r}")


def run_batch(inp):
    """fit_val_batch: one call fits a GRID of trials on shared (Xtr,ytr,Xva,yva). Each trial is isolated
    (a bad trial returns an error stub, never failing the grid). Amortizes cold-start + data upload."""
    shared = {k: inp[k] for k in ("task_type", "Xtr", "ytr", "Xva", "yva") if k in inp}
    results = []
    for t in inp.get("trials", []):
        one = dict(shared, op="fit_val", family=t["family"], params=t.get("params"),
                   seed=int(t.get("seed", 0)), n_classes=inp.get("n_classes"))
        try:
            out = run(one)
            out["content_id"] = t.get("content_id")
            results.append(out)
        except Exception as e:  # noqa: BLE001  isolate per-trial failure
            results.append({"content_id": t.get("content_id"), "family": t["family"],
                            "error": f"{type(e).__name__}: {str(e)[:160]}"})
    return {"results": results, "device": _device()}


def handler(job):
    """RunPod serverless entrypoint: job = {'input': {...}}. Dispatches single or batch ops."""
    try:
        inp = job.get("input", {})
        return run_batch(inp) if inp.get("op") == "fit_val_batch" else run(inp)
    except Exception as e:  # noqa: BLE001  surface a clean error to the caller
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


if __name__ == "__main__":
    # In the RunPod image this starts the serverless loop; locally it is import-only for smoke tests.
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except Exception as e:  # noqa: BLE001
        print(f"runpod serverless not started (expected outside the worker image): {e}")
