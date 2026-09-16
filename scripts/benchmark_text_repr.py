"""#2 -- THE LAW, REPRODUCED ON TEXT. Two panels, byte-identical FDR discipline to B1->#5.

The seven vision phases established one law: given a fixed representation, nothing AUTHORED on top moves the
metric (search 0/5, novel classifiers 0/5, fine-tuning 0/10, featurizers 0/10); the ONLY lever that moves it is
CHANGING the representation, and within the winning family scale pays. This script tests whether that law is a
vision artifact or a general one, on TEXT (20-Newsgroups confusable pairs over frozen sentence encoders).

PANEL B -- "representation is the lever": each frozen sentence encoder vs the WEAK lexical baseline (TF-IDF+LSA),
one-sided exact McNemar on IDENTICAL sealed rows, BH-FDR(0.1) across the suite. Survivors with positive lift =
the representation moved the metric. The e5 family (small->base->large) is a clean within-family scale ladder --
the text analog of DINOv2 S->L->g.

PANEL A -- "given a fixed representation, nothing authored on top beats a tuned model-search": on the CHAMPION
encoder's frozen embeddings, the strong baseline head is the SAME tuned random search over GBM/RF/ET/logreg/knn
(benchmark_vision_transfer._random_search_best). Authored heads NOT contained in that search -- a tuned RBF-SVM,
a tuned MLP, a cosine-prototype classifier, and whitening+logreg -- are each compared head-to-head against the
strong baseline on identical sealed rows, BH-FDR(0.1). Survivors = authoring beat the tuned head.

Expectation if the law generalizes: Panel B has survivors; Panel A has ZERO. Every arm uses the identical
sealed rows (derived once from the lexical baseline), the same val-selected fitting, and the frozen
Clopper-Pearson lower bound. Run: `python scripts/benchmark_text_repr.py`.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neural_network import MLPClassifier            # noqa: E402
from sklearn.pipeline import make_pipeline                  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from sklearn.svm import SVC                                 # noqa: E402

import scripts.benchmark_vision_transfer as B1              # noqa: E402
import scripts.repr_arena_text as T                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.1
OUT_NAME = os.environ.get("ATTESTRA_OUT", "BENCHMARK_TEXT_REPR_RESULT.json")


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in ("vectorforge/science.py", "vfplatform/sealed.py")}


# ---- authored heads (val-selected, fit on train rows only, scored on sealed) -- the "authoring" arms --------
def _svm_rbf(Xtr, ytr, Xva, yva):
    best, bv = None, -1.0
    for C in (1.0, 10.0, 100.0):
        for gamma in ("scale", 0.01, 0.1):
            est = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf"))
            est.fit(Xtr, ytr)
            v = est.score(Xva, yva)
            if v > bv:
                bv, best = v, est
    return best


def _mlp(Xtr, ytr, Xva, yva):
    best, bv = None, -1.0
    for hidden in ((256,), (128, 64)):
        for alpha in (1e-4, 1e-3):
            est = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=500,
                                              early_stopping=True, random_state=0))
            est.fit(Xtr, ytr)
            v = est.score(Xva, yva)
            if v > bv:
                bv, best = v, est
    return best


def _cosine_prototype(Xtr, ytr, Xva, yva):
    class _Proto:
        def fit(self, X, y):
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            self.protos = np.stack([Xn[y == c].mean(0) for c in (0, 1)])
            self.protos /= (np.linalg.norm(self.protos, axis=1, keepdims=True) + 1e-9)
            return self

        def predict(self, X):
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            return (Xn @ self.protos.T).argmax(1)
    return _Proto().fit(Xtr, ytr)


def _whiten_logreg(Xtr, ytr, Xva, yva):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    best, bv = None, -1.0
    for C in (0.1, 1.0, 10.0):
        k = min(128, Xtr.shape[1] - 1, Xtr.shape[0] - 1)
        est = make_pipeline(PCA(n_components=k, whiten=True, random_state=0),
                            LogisticRegression(C=C, max_iter=2000))
        est.fit(Xtr, ytr)
        v = est.score(Xva, yva)
        if v > bv:
            bv, best = v, est
    return best


AUTHORED = [("svm_rbf", "tuned RBF-SVM", _svm_rbf), ("mlp", "tuned MLP", _mlp),
            ("cos_proto", "cosine-prototype", _cosine_prototype),
            ("whiten_logreg", "whitening + logreg", _whiten_logreg)]


def main():
    arena = T.TwentyNewsArena(smoke=int(os.environ.get("ATTESTRA_SMOKE", "0")))
    tasks = arena.tasks
    print(f"frozen certifier: {_frozen_hashes()}")
    print(f"tasks={len(tasks)} per_class={arena.per_class} encoders={[e.tag for e in T.REGISTRY]}\n")

    # ---- PANEL B: each encoder vs the weak lexical baseline -------------------------------------------------
    base = arena.measure(T.BASELINE_TAG)
    base_c = {t: base[t].sealed_correct for t in tasks}
    panelB, enc_pool = {}, {}
    for enc in T.REGISTRY:
        meas = arena.measure(enc.tag)
        pvals, lifts, per = [], [], []
        pooled = []
        for t in tasks:
            p = arena.mcnemar(meas[t].sealed_correct, base_c[t])
            lift = meas[t].acc - base[t].acc
            per.append({"task": t, "acc": round(meas[t].acc, 4), "base_acc": round(base[t].acc, 4),
                        "lift": round(lift, 4), "p_gt_base": round(p, 4)})
            pvals.append(p); lifts.append(lift)
            pooled.extend(list(meas[t].sealed_correct))
        rej = set(arena.bh(pvals, ALPHA))
        survivors = [tasks[i] for i in range(len(tasks)) if i in rej and lifts[i] > 0]
        panelB[enc.tag] = {
            "family": enc.family, "scale_rank": enc.scale_rank, "params_m": enc.params_m,
            "survivors_vs_lexical": survivors, "n": len(tasks),
            "mean_lift_vs_lexical": round(float(np.mean(lifts)), 4),
            "pooled_sealed_acc": round(float(np.mean(pooled)), 4),
            "pooled_sealed_lb": arena.lower_bound(pooled), "per_task": per}
        enc_pool[enc.tag] = pooled
        print(f"[B] {enc.tag:10} ({enc.family:7}) vs lexical: {len(survivors)}/{len(tasks)} FDR  "
              f"mean_lift {panelB[enc.tag]['mean_lift_vs_lexical']:+.3f}  "
              f"pooled_acc {panelB[enc.tag]['pooled_sealed_acc']:.3f}  lb {panelB[enc.tag]['pooled_sealed_lb']:.3f}")

    # champion = best NON-lexical encoder by pooled sealed lower bound (the policy's tie-break criterion)
    neural = [e.tag for e in T.REGISTRY if e.family != "lexical"]
    champ = max(neural, key=lambda tg: panelB[tg]["pooled_sealed_lb"])
    print(f"\nchampion by pooled sealed lower bound: {champ} "
          f"(lb {panelB[champ]['pooled_sealed_lb']:.3f})\n")

    # ---- PANEL A: on the CHAMPION encoder, authored heads vs the tuned model-search ------------------------
    strong = arena.measure(champ)             # the tuned random-search head on the champion embeddings
    strong_c = {t: strong[t].sealed_correct for t in tasks}
    panelA = {}
    for tag, label, fn in AUTHORED:
        pvals, lifts, per = [], [], []
        for t in tasks:
            sp = arena._task_split(t)
            emb, y = arena._emb(champ, t), sp["y"]
            est = fn(emb[sp["tr"]], y[sp["tr"]], emb[sp["val"]], y[sp["val"]])
            c = B1._correct(est, emb[sp["test"]], y[sp["test"]])
            acc = float(np.mean(c))
            p = arena.mcnemar(c, strong_c[t])
            lift = acc - strong[t].acc
            per.append({"task": t, "authored_acc": round(acc, 4), "strong_acc": round(strong[t].acc, 4),
                        "lift": round(lift, 4), "p_gt_strong": round(p, 4)})
            pvals.append(p); lifts.append(lift)
        rej = set(arena.bh(pvals, ALPHA))
        survivors = [tasks[i] for i in range(len(tasks)) if i in rej and lifts[i] > 0]
        panelA[tag] = {"label": label, "survivors_vs_strong": survivors, "n": len(tasks),
                       "mean_lift_vs_strong": round(float(np.mean(lifts)), 4), "per_task": per}
        print(f"[A] {label:22} vs tuned-search on {champ}: {len(survivors)}/{len(tasks)} FDR  "
              f"mean_lift {panelA[tag]['mean_lift_vs_strong']:+.3f}")

    out = {"arena": "20newsgroups-confusable-pairs", "per_class": arena.per_class, "alpha": ALPHA,
           "baseline": T.BASELINE_TAG, "champion": champ, "tasks": tasks,
           "panelB_representation_lever": panelB, "panelA_authoring_on_fixed_rep": panelA,
           "frozen_hashes": _frozen_hashes()}
    dst = os.path.join(ROOT, "docs", OUT_NAME)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}\nfrozen (post): {_frozen_hashes()}")


if __name__ == "__main__":
    main()
