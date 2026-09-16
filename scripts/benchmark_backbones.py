"""B2-repr -- IS THE REPRESENTATION THE LEVER? (does a stronger/different backbone beat resnet18 emb-strong?)

Three independent results now agree that, GIVEN a fixed representation, neither the search cycle (audit 0/4,
B1 0/5) nor LLM-authored novel methods (B2 0/5) beat a tuned GBM. The single lever that DID move the metric was
the REPRESENTATION: B1's emb-strong (frozen resnet18 embeddings) beat raw pixels 5/5 FDR (+0.14-0.22). This
benchmark attacks that lever directly: on the SAME B1 arena (same five CIFAR-10 tasks, same per_class, same
identical sealed test), does a STRONGER or DIFFERENT frozen backbone beat the resnet18 emb-strong baseline?

The comparator is exactly B1's deployed representation: resnet18 (resize=112) + the strong baseline (tuned GBM
+ random search over the catalog). The challenger backbones, each frozen and each scored by the SAME strong
baseline on its OWN embeddings:

  resnet50         : a deeper ImageNet-supervised CNN (2048-d).
  clip_vitb32      : OpenAI CLIP image tower -- a LANGUAGE-ALIGNED representation (512-d).
  dinov2_vits14    : a SELF-SUPERVISED ViT (DINOv2, 384-d) -- the strongest frozen-feature transfer family.

For every backbone we fit emb-strong on the IDENTICAL train rows, score the IDENTICAL sealed rows, and report
sealed accuracy + the FROZEN Clopper-Pearson lower bound (science.clopper_pearson_lower, read-only). Each
challenger is paired against resnet18 by one-sided exact McNemar (challenger > resnet18) on the sealed rows,
then Benjamini-Hochberg(0.1) across the five-task suite, PER backbone. The frontier claim is a challenger with
an FDR-surviving positive lift over resnet18 emb-strong. The frozen certifier is the sole promoter, untouched;
this script only MEASURES and reports honest results (positive or negative) as first-class.
"""
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorforge import science                               # noqa: E402
from vfplatform.battery import mcnemar_pvalue, benjamini_hochberg  # noqa: E402
from vfplatform.featurizers import (ImageFeaturizer, ResnetBackbone,  # noqa: E402
                                    ClipImageBackbone, TimmBackbone)
import scripts.benchmark_vision_transfer as B1                # noqa: E402

PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "500"))
ALPHA = 0.05
BASELINE = os.environ.get("ATTESTRA_BASELINE", "resnet18")
ROSTER = os.environ.get("ATTESTRA_ROSTER", "base")
OUT_NAME = os.environ.get("ATTESTRA_OUT", "BENCHMARK_BACKBONES_RESULT.json")

# Two rosters. "base" (B2-repr): does a stronger/different backbone beat resnet18? "big" (B2-repr-2): having
# established CLIP-B/DINOv2-S win, does SCALING the winning inductive bias (ViT-L) or a stronger language-image
# objective (SigLIP) keep climbing, or has this arena hit its frozen-feature ceiling? "big"'s baseline is the
# prior champion clip_vitb32 (set ATTESTRA_BASELINE=clip_vitb32), so each L-scale challenger is paired vs it.
_SPECS = {
    "base": [
        ("resnet18", lambda: ResnetBackbone(arch="resnet18", resize=112, batch_size=64, pretrained=True)),
        ("resnet50", lambda: ResnetBackbone(arch="resnet50", resize=224, batch_size=64, pretrained=True)),
        ("clip_vitb32", lambda: ClipImageBackbone(model_name="ViT-B-32-quickgelu", pretrained="openai",
                                                   batch_size=64)),
        ("dinov2_vits14", lambda: TimmBackbone(model_name="vit_small_patch14_dinov2.lvd142m",
                                               pretrained=True, batch_size=32, img_size=224)),
    ],
    "big": [
        ("resnet18", lambda: ResnetBackbone(arch="resnet18", resize=112, batch_size=64, pretrained=True)),
        ("clip_vitb32", lambda: ClipImageBackbone(model_name="ViT-B-32-quickgelu", pretrained="openai",
                                                   batch_size=64)),
        ("dinov2_vits14", lambda: TimmBackbone(model_name="vit_small_patch14_dinov2.lvd142m",
                                               pretrained=True, batch_size=32, img_size=224)),
        ("clip_vitl14", lambda: ClipImageBackbone(model_name="ViT-L-14-quickgelu", pretrained="openai",
                                                  batch_size=16)),
        ("dinov2_vitl14", lambda: TimmBackbone(model_name="vit_large_patch14_dinov2.lvd142m",
                                               pretrained=True, batch_size=8, img_size=224)),
        ("siglip_vitl16", lambda: ClipImageBackbone(model_name="ViT-L-16-SigLIP-256", pretrained="webli",
                                                    batch_size=16)),
    ],
}


def _embed(backbone, task_imgs):
    return ImageFeaturizer(backbone=backbone, backbone_dim=backbone.feat_dim).transform(task_imgs)


def _sealed_split(base_emb, y, seed=0):
    """Derive B1's IDENTICAL sealed test + the loop's internal train/val split FROM THE BASELINE (resnet18)
    embeddings -- exactly as B1 does -- then reuse the resulting row ids for every backbone so the McNemar
    pairing is on identical sealed rows. (The split must run on real, distinct embeddings: make_splits'
    near-duplicate dedup would collapse constant placeholder rows.)"""
    n = len(y); rids = np.arange(n)
    rng = np.random.RandomState(seed + 7)
    test_ids = []
    for c in (0, 1):
        ci = rids[y == c]; rng.shuffle(ci); test_ids += list(ci[: int(0.3 * len(ci))])
    test_ids = set(int(i) for i in test_ids)
    trainval_ids = [int(i) for i in rids if int(i) not in test_ids]
    emb_recs = B1._records(base_emb, y, rids)
    tv_recs = [emb_recs[i] for i in trainval_ids]
    tr2, va2, te2 = B1._split(tv_recs, "target", "text", seed, False)
    tr_pool_ids = [r["rid"] for r in (tr2 + te2)]
    val_ids = [r["rid"] for r in va2]
    return tr_pool_ids, val_ids, sorted(test_ids)


def _emb_strong_correct(emb, y, tr_ids, val_ids, test_ids, seed=0):
    """emb-strong on a backbone's embeddings: tuned GBM + random search selected on val, scored on the sealed
    rows. Returns (sealed correctness vector, sealed accuracy)."""
    def _slice(ids):
        ids = list(ids); return emb[ids], y[ids]
    Xtr, ytr = _slice(tr_ids); Xva, yva = _slice(val_ids); Xte, yte = _slice(test_ids)
    est = B1._random_search_best(np.random.RandomState(seed), Xtr, ytr, Xva, yva)
    correct = B1._correct(est, Xte, yte)
    return correct, float(np.mean(correct))


def _baseline_cache(imgs, labels, baseline_bb, per_class=PER_CLASS, seed=0):
    """Phase A (ONLY the baseline backbone resident): per task, compute baseline embeddings, derive B1's
    identical split ONCE, and record baseline emb-strong correctness/accuracy. The split row-ids are reused
    verbatim for every challenger so all McNemar pairing is on identical sealed rows."""
    cache = {}
    for a, b in B1.SUITE:
        name = f"{a}_vs_{b}"
        task_imgs, y = B1._task_data(imgs, labels, a, b, per_class, seed)
        base_emb = _embed(baseline_bb, task_imgs)
        tr_ids, val_ids, test_ids = _sealed_split(base_emb, y, seed)
        c, acc = _emb_strong_correct(base_emb, y, tr_ids, val_ids, test_ids, seed)
        cache[name] = {"y": y, "tr": tr_ids, "val": val_ids, "test": test_ids,
                       "base_c": c, "acc": round(acc, 4), "lb": B1._lb(c), "dim": int(base_emb.shape[1])}
        print(f"  [A] {name:18} {BASELINE}={cache[name]['acc']}(lb{cache[name]['lb']}) "
              f"n_test={len(test_ids)}")
    return cache


def _frozen_hashes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()[:8]
            for f in ("vectorforge/science.py", "vfplatform/sealed.py")}


def _summarize(rows, challengers, errs):
    ok = [r for r in rows if r.get("p_gt_base")]
    summary = {}
    for c in challengers:
        if c in errs:
            continue
        ps = [r["p_gt_base"][c] for r in ok if c in r["p_gt_base"]]
        rel = [r for r in ok if c in r["p_gt_base"]]
        if not ps:
            continue
        rej = set(benjamini_hochberg(ps, alpha=0.1))
        wins = [rel[i]["name"] for i in range(len(rel)) if i in rej and rel[i]["lift_vs_base"][c] > 0]
        summary[c] = {"survivors": wins, "n": len(rel),
                      "mean_lift": round(float(np.mean([r["lift_vs_base"][c] for r in rel])), 4)}
    return summary


def _write_out(rows_list, challengers, errs, dst):
    out = {"roster": ROSTER, "per_class": PER_CLASS, "baseline": BASELINE, "backbone_errors": errs,
           "rows": rows_list, "summary": _summarize(rows_list, challengers, errs),
           "frozen_hashes": _frozen_hashes()}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    print(f"[B2-repr] roster={ROSTER} | per_class={PER_CLASS} | baseline={BASELINE} | out={OUT_NAME}")
    imgs, labels = B1._load_cifar()
    specs = _SPECS[ROSTER]
    names = [n for n, _ in specs]
    factory_of = dict(specs)
    if BASELINE not in factory_of:
        raise SystemExit(f"comparator backbone {BASELINE!r} not in roster {ROSTER!r}")
    challengers = [n for n in names if n != BASELINE]
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", OUT_NAME)

    # Phase A: baseline only resident -> per-task split + baseline correctness.
    print(f"[A] baseline {BASELINE} resident; deriving splits + baseline correctness")
    base_bb = factory_of[BASELINE]()
    print(f"  backbone {BASELINE:14} dim={base_bb.feat_dim}")
    cache = _baseline_cache(imgs, labels, base_bb)
    del base_bb; gc.collect()

    rows = {name: {"name": name, "n_test": len(cache[name]["test"]),
                   "acc": {BASELINE: cache[name]["acc"]}, "lb": {BASELINE: cache[name]["lb"]},
                   "dim": {BASELINE: cache[name]["dim"]}, "lift_vs_base": {}, "p_gt_base": {}, "secs": {}}
            for name in (f"{a}_vs_{b}" for a, b in B1.SUITE)}
    errs = {}

    # Phase B: ONE challenger backbone resident at a time (7GB box; three ViT-L towers cannot coexist).
    for name in challengers:
        t0 = time.time()
        try:
            bb = factory_of[name]()
        except Exception as e:  # noqa: BLE001
            errs[name] = f"{type(e).__name__}: {str(e)[:140]}"
            print(f"  [B] backbone {name:14} UNAVAILABLE: {errs[name]}")
            continue
        print(f"  [B] backbone {name:14} dim={bb.feat_dim} resident")
        for a, b in B1.SUITE:
            tn = f"{a}_vs_{b}"
            task_imgs, y = B1._task_data(imgs, labels, a, b, PER_CLASS, 0)
            emb = _embed(bb, task_imgs)
            ci = cache[tn]
            c, acc = _emb_strong_correct(emb, y, ci["tr"], ci["val"], ci["test"], 0)
            rows[tn]["acc"][name] = round(acc, 4)
            rows[tn]["lb"][name] = B1._lb(c)
            rows[tn]["dim"][name] = int(emb.shape[1])
            rows[tn]["lift_vs_base"][name] = round(acc - ci["acc"], 4)
            rows[tn]["p_gt_base"][name] = mcnemar_pvalue(c, ci["base_c"])
            del emb, task_imgs
        secs = round(time.time() - t0, 1)
        for a, b in B1.SUITE:
            rows[f"{a}_vs_{b}"]["secs"][name] = secs
        print(f"      {name} done all tasks [{secs}s]: " +
              " ".join(f"{a}_vs_{b}={rows[f'{a}_vs_{b}']['acc'][name]}"
                       f"({rows[f'{a}_vs_{b}']['lift_vs_base'][name]:+.3f},"
                       f"p{rows[f'{a}_vs_{b}']['p_gt_base'][name]:.2f})" for a, b in B1.SUITE))
        del bb; gc.collect()
        # crash-safe checkpoint: persist completed backbones after each (the ViT-L run is ~3h on this box).
        _write_out([rows[f"{a}_vs_{b}"] for a, b in B1.SUITE], challengers, errs, dst)

    rows_list = [rows[f"{a}_vs_{b}"] for a, b in B1.SUITE]
    out = _write_out(rows_list, challengers, errs, dst)
    for c, s in out["summary"].items():
        print(f"\n=== {c} vs {BASELINE}: BH-FDR(0.1) survivors with positive lift: "
              f"{len(s['survivors'])}/{s['n']} -> {s['survivors']}  (mean lift {s['mean_lift']:+})")
    print(f"\nwrote {dst}\nfrozen: {out['frozen_hashes']}")


if __name__ == "__main__":
    main()
