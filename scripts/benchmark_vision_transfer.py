"""B1 -- A HEADROOM ARENA WITH AN HONEST STRONG BASELINE (the test the tabular suite could not provide).

The audit's decisive finding: on the tabular OpenML suite the full recursive cycle does NOT beat a tuned
HistGradientBoosting baseline -- there is no headroom, so "the system adds value" is unfalsifiable there.
This arena fixes that. On confusable CIFAR-10 binary tasks a strong baseline operating on RAW PIXELS is
provably far from the ceiling, while a frozen ImageNet backbone (resnet18) + a certifiable head is strong.
That gap is exactly the headroom the autoresearcher must learn to capture (via transfer/representation,
the one lever the audit's own commercial examples -- biopsy, call-ender -- depend on).

To avoid the prior session's STRAWMAN (beating logistic), every comparison here is paired on an IDENTICAL
sealed test against a STRONG baseline (tuned GBM + random search over the catalog), and three arms are run:

  A1  raw-strong   : tuned GBM + random search over the catalog on RAW PIXELS        (no transfer, no cycle)
  A2  emb-strong   : the SAME strong baseline on FROZEN resnet18 EMBEDDINGS           (transfer, no cycle)
  C   cycle-emb    : run_goal_loop (the full system) on the SAME embeddings           (transfer + the cycle)

For each task we report each arm's sealed-test accuracy + the FROZEN Clopper-Pearson lower bound
(science.clopper_pearson_lower, read-only use of the frozen certifier -- the files are never edited), and
paired one-sided exact McNemar of C>A1, C>A2, A2>A1 on the sealed rows, then Benjamini-Hochberg across the
suite. This answers two honest questions the tabular suite could not:

  (i)  is there headroom beyond a strong RAW baseline?           -> A2/C vs A1
  (ii) given the right representation, does the CYCLE beat a strong baseline? -> C vs A2

The frozen certifier is the sole promoter and is untouched. This script never weakens a gate; it only
MEASURES, and reports honest negatives as first-class results.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,  # noqa: E402
                              ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier            # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.pipeline import make_pipeline                    # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402

from vectorforge import science                               # noqa: E402
from vfplatform.loop import run_goal_loop, _split             # noqa: E402
from vfplatform.harness import harness_for                    # noqa: E402
from vfplatform.battery import mcnemar_pvalue, benjamini_hochberg  # noqa: E402
from vfplatform.featurizers import ImageFeaturizer, ResnetBackbone  # noqa: E402

DATA_DIR = os.environ.get("ATTESTRA_DATA", "/home/ubuntu/data")
CIFAR = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
         "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
# Confusable pairs -> genuine headroom for a raw-pixel learner; a suite gives BH-FDR something to control.
SUITE = [("cat", "dog"), ("automobile", "truck"), ("deer", "horse"),
         ("airplane", "ship"), ("bird", "frog")]
PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "500"))
ALPHA = 0.05


def _load_cifar():
    import torchvision as tv
    ds = tv.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    imgs = ds.data                      # (50000, 32, 32, 3) uint8
    labels = np.array(ds.targets)
    return imgs, labels


def _task_data(imgs, labels, cls_a, cls_b, per_class, seed=0):
    rng = np.random.RandomState(seed)
    out_imgs, out_y = [], []
    for cls, y in [(cls_a, 0), (cls_b, 1)]:
        idx = np.where(labels == CIFAR[cls])[0]
        rng.shuffle(idx)
        for i in idx[:per_class]:
            out_imgs.append(imgs[i]); out_y.append(y)
    order = rng.permutation(len(out_y))
    return [out_imgs[i] for i in order], np.array([out_y[i] for i in order])


def _records(feat_matrix, y, rids):
    width = len(str(feat_matrix.shape[1]))
    recs = []
    for row, yi, rid in zip(feat_matrix, y, rids):
        recs.append({"features": {f"e{str(k).zfill(width)}": float(v) for k, v in enumerate(row)},
                     "target": str(int(yi)), "rid": int(rid)})
    return recs


def _xy(recs):
    keys = sorted(recs[0]["features"].keys())
    X = np.array([[r["features"][k] for k in keys] for r in recs], dtype=float)
    y = np.array([int(r["target"]) for r in recs])
    return X, y


def _random_search_best(rng, Xtr, ytr, Xva, yva, k=15):
    best, best_v = None, -1.0
    for _ in range(k):
        fam = rng.choice(["gbm", "rf", "et", "logreg", "knn"])
        if fam == "gbm":
            est = HistGradientBoostingClassifier(learning_rate=float(rng.choice([0.05, 0.1, 0.2])),
                                                 max_iter=int(rng.choice([100, 300])),
                                                 max_depth=rng.choice([None, 6, 12]), random_state=0)
        elif fam == "rf":
            est = RandomForestClassifier(n_estimators=int(rng.choice([200, 400])),
                                         max_depth=rng.choice([None, 12]),
                                         max_features=rng.choice(["sqrt", 0.5]), random_state=0, n_jobs=1)
        elif fam == "et":
            est = ExtraTreesClassifier(n_estimators=int(rng.choice([200, 400])),
                                       max_features=rng.choice(["sqrt", 0.5]), random_state=0, n_jobs=1)
        elif fam == "logreg":
            est = make_pipeline(StandardScaler(),
                                LogisticRegression(C=float(rng.choice([0.1, 1.0, 10.0])), max_iter=2000))
        else:
            est = make_pipeline(StandardScaler(),
                                KNeighborsClassifier(n_neighbors=int(rng.choice([5, 11, 21])), n_jobs=1))
        try:
            est.fit(Xtr, ytr)
            v = est.score(Xva, yva)
        except Exception:  # noqa: BLE001
            continue
        if v > best_v:
            best_v, best = v, est
    # always include a tuned-GBM default so the baseline is never weaker than one strong model
    gbm = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    if gbm.score(Xva, yva) > best_v or best is None:
        best = gbm
    return best


def _correct(est, Xte, yte):
    return (est.predict(Xte) == yte).astype(int).tolist()


def _lb(correct):
    k, n = int(sum(correct)), len(correct)
    return round(science.clopper_pearson_lower(k, n, ALPHA), 4)


def run_task(imgs, labels, cls_a, cls_b, backbone, per_class=PER_CLASS, seed=0):
    name = f"{cls_a}_vs_{cls_b}"
    task_imgs, y = _task_data(imgs, labels, cls_a, cls_b, per_class, seed)
    n = len(y)
    rids = np.arange(n)
    raw = np.array([im.reshape(-1).astype(float) / 255.0 for im in task_imgs])   # 3072-d raw pixels
    emb = ImageFeaturizer(backbone=backbone, backbone_dim=backbone.feat_dim).transform(task_imgs)

    emb_recs = _records(emb, y, rids)
    # outer sealed test = a fresh stratified 30% held out BEFORE the loop sees anything; identical across arms
    rng = np.random.RandomState(seed + 7)
    test_ids = []
    for c in (0, 1):
        ci = rids[y == c]; rng.shuffle(ci); test_ids += list(ci[: int(0.3 * len(ci))])
    test_ids = set(int(i) for i in test_ids)
    trainval_ids = [int(i) for i in rids if int(i) not in test_ids]

    def _slice(ids, M):
        ids = list(ids)
        return M[ids], y[ids]

    # re-derive the loop's internal split WITHIN trainval so baselines use the loop's exact train/val
    tv_emb_recs = [emb_recs[i] for i in trainval_ids]
    tr2, va2, te2 = _split(tv_emb_recs, "target", "text", seed, False)
    tr_pool_ids = [r["rid"] for r in (tr2 + te2)]
    val_ids = [r["rid"] for r in va2]
    test_ids_l = sorted(test_ids)

    Xtr_raw, ytr = _slice(tr_pool_ids, raw); Xva_raw, yva = _slice(val_ids, raw)
    Xtr_emb, _ = _slice(tr_pool_ids, emb);   Xva_emb, _ = _slice(val_ids, emb)
    Xte_raw, yte = _slice(test_ids_l, raw);  Xte_emb, _ = _slice(test_ids_l, emb)

    rs = np.random.RandomState(seed)
    a1 = _random_search_best(rs, Xtr_raw, ytr, Xva_raw, yva)
    a2 = _random_search_best(rs, Xtr_emb, ytr, Xva_emb, yva)
    a1_c = _correct(a1, Xte_raw, yte)
    a2_c = _correct(a2, Xte_emb, yte)

    # arm C: the full cycle on embeddings; SAME sealed test via test_records
    tv_recs = [emb_recs[i] for i in trainval_ids]
    test_recs = [emb_recs[i] for i in test_ids_l]
    res = run_goal_loop(tv_recs, f"classify {name}", harness=harness_for("tabular", "binary"),
                        target_key="target", labels=["0", "1"], threshold=0.5, metric="accuracy",
                        seeds=(0,), min_test_n=80, seed=seed, experiment=f"vt-{name}",
                        objective="maximize", max_rounds=6, budget_experiments=48,
                        llm_enabled=False, llm_propose=False, test_records=test_recs)
    c_c = res.cand_locked_correct
    if not c_c:
        return {"name": name, "excluded": True, "decision": res.decision, "n_test": len(yte),
                "acc_raw_strong": round(np.mean(a1_c), 4), "lb_raw_strong": _lb(a1_c),
                "acc_emb_strong": round(np.mean(a2_c), 4), "lb_emb_strong": _lb(a2_c)}
    # align C to the same row order as a1_c/a2_c (the loop keeps test_records order)
    if len(c_c) != len(yte):
        return {"name": name, "error": f"cycle len {len(c_c)} != test {len(yte)}"}
    return {"name": name, "n_test": len(yte), "winner": (res.winner.family if res.winner else None),
            "acc_raw_strong": round(np.mean(a1_c), 4), "lb_raw_strong": _lb(a1_c),
            "acc_emb_strong": round(np.mean(a2_c), 4), "lb_emb_strong": _lb(a2_c),
            "acc_cycle_emb": round(np.mean(c_c), 4), "lb_cycle_emb": _lb(c_c),
            "lift_cycle_vs_raw": round(np.mean(c_c) - np.mean(a1_c), 4),
            "lift_cycle_vs_embstrong": round(np.mean(c_c) - np.mean(a2_c), 4),
            "lift_embstrong_vs_raw": round(np.mean(a2_c) - np.mean(a1_c), 4),
            "p_cycle_gt_raw": mcnemar_pvalue(c_c, a1_c),
            "p_cycle_gt_embstrong": mcnemar_pvalue(c_c, a2_c),
            "p_embstrong_gt_raw": mcnemar_pvalue(a2_c, a1_c)}


def main():
    print(f"[B1] vision-transfer headroom arena | per_class={PER_CLASS} | data={DATA_DIR}")
    imgs, labels = _load_cifar()
    backbone = ResnetBackbone(arch="resnet18", resize=112, batch_size=64, pretrained=True)
    rows = []
    for a, b in SUITE:
        t0 = time.time()
        try:
            r = run_task(imgs, labels, a, b, backbone)
        except Exception as e:  # noqa: BLE001
            r = {"name": f"{a}_vs_{b}", "error": str(e)[:160]}
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        if r.get("excluded"):
            print(f"  {r['name']:18} EXCLUDED ({r['decision']}) raw_strong={r['acc_raw_strong']} "
                  f"emb_strong={r['acc_emb_strong']}")
        elif "error" in r:
            print(f"  {r['name']:18} ERROR {r['error']}")
        else:
            print(f"  {r['name']:18} raw_strong={r['acc_raw_strong']}(lb{r['lb_raw_strong']}) "
                  f"emb_strong={r['acc_emb_strong']}(lb{r['lb_emb_strong']}) "
                  f"cycle={r['acc_cycle_emb']}(lb{r['lb_cycle_emb']}) | "
                  f"C>raw {r['lift_cycle_vs_raw']:+.3f}(p{r['p_cycle_gt_raw']:.1e}) "
                  f"C>embGBM {r['lift_cycle_vs_embstrong']:+.3f}(p{r['p_cycle_gt_embstrong']:.1e}) "
                  f"embGBM>raw {r['lift_embstrong_vs_raw']:+.3f}(p{r['p_embstrong_gt_raw']:.1e}) [{r['secs']}s]")
    ok = [r for r in rows if "p_cycle_gt_raw" in r]
    summary = {}
    for key, liftk, label in [("p_cycle_gt_raw", "lift_cycle_vs_raw", "CYCLE(emb) vs RAW-STRONG"),
                              ("p_cycle_gt_embstrong", "lift_cycle_vs_embstrong", "CYCLE(emb) vs EMB-STRONG"),
                              ("p_embstrong_gt_raw", "lift_embstrong_vs_raw", "EMB-STRONG vs RAW-STRONG")]:
        ps = [r[key] for r in ok]
        rej = set(benjamini_hochberg(ps, alpha=0.1)) if ps else set()
        wins = [ok[i]["name"] for i in range(len(ok)) if i in rej and ok[i][liftk] > 0]
        summary[label] = {"survivors": wins, "n": len(ok)}
        print(f"\n=== {label}: BH-FDR(0.1) survivors with positive lift: {len(wins)}/{len(ok)} -> {wins}")
    print("\nHONEST READING:")
    print("  EMB-STRONG vs RAW-STRONG survivors  -> the arena HAS headroom transfer captures (the point of B1).")
    print("  CYCLE vs EMB-STRONG survivors       -> the cycle adds value BEYOND a strong baseline given the")
    print("                                         representation (the frontier claim). Expected ~0 here, as on")
    print("                                         tabular -- which is why B2 (novel-method authoring) is next.")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fh = {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()[:8]
          for f in ("vectorforge/science.py", "vfplatform/sealed.py")}
    out = {"per_class": PER_CLASS, "rows": rows, "summary": summary, "frozen_hashes": fh}
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "BENCHMARK_VISION_TRANSFER_RESULT.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
