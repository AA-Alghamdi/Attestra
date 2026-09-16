"""Phase #4 -- AUTHORED FEATURIZER vs the FROZEN CHAMPION: can COMBINING/TRANSFORMING frozen representations beat
the single best frozen representation (DINOv2-L) on the hard FGVC-Aircraft arena?

The arc so far establishes one law across two arenas: given a representation, neither the search cycle (B1 0/5),
nor LLM-authored *classifiers* (B2 0/5), nor end-to-end fine-tuning (#3 0/10 vs the best frozen rep) beats a
tuned GBM head -- only CHANGING the representation moves the metric (CLIP/DINOv2 over resnet18: 5/5; DINOv2-L
over CLIP-B on the hard arena: 2/10). That leaves exactly ONE untested form of "authoring": authoring a
REPRESENTATION (a featurizer), not a classifier. DINOv2-L (self-supervised, fine-grained) and CLIP-B
(language-aligned) plausibly encode COMPLEMENTARY structure, so a featurizer that fuses/transforms them could
beat either alone -- which would be the first new authored lever since transfer itself.

Discipline is identical to #3: the train/val/test row-ids are derived ONCE from the CLIP-B baseline embeddings
(`benchmark_backbones._sealed_split`) and reused verbatim, so all McNemar pairing is on identical sealed rows;
every arm uses the SAME strong tuned-GBM head (`benchmark_backbones._emb_strong_correct`) so ONLY the
representation differs; quality is the frozen Clopper-Pearson lower bound (read-only). Arms:

  clip_vitb32   : frozen CLIP ViT-B/32 (512-d)             -> weak reference
  dinov2_vitl14 : frozen DINOv2 ViT-L/14 (1024-d)          -> the CHAMPION (the baseline we must beat)
  fuse_concat   : concat[DINOv2-L || CLIP-B] (1536-d)      -> authored featurizer: fuse two frozen biases
  fuse_pls      : supervised PLS(32) of the concat, fit on TRAIN rows only -> authored featurizer: a LEARNED
                  low-dim representation of the fused space (no val/test leakage; PLS sees train labels only)

Comparisons (one-sided exact McNemar on identical sealed rows, then BH-FDR(0.1) across the 10-pair suite):
(fuse_concat > dinov2), (fuse_pls > dinov2), plus (fuse_* > clip) for context. If a fusion arm FDR-survives over
DINOv2-L, fusing frozen reps is a real authored lever -> escalate to LLM-authored featurizers. If it ties (as
authored classifiers did), the ceiling on low-shot fine-grained vision is the single best pretrained encoder
itself, and the frontier move is a better/bigger encoder -- not anything authored on top.
"""
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.battery import mcnemar_pvalue, benjamini_hochberg  # noqa: E402
import scripts.benchmark_vision_transfer as B1                     # noqa: E402
import scripts.benchmark_backbones as BB                           # noqa: E402
import scripts.benchmark_aircraft as A                             # noqa: E402

PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "100"))
SMOKE = int(os.environ.get("ATTESTRA_SMOKE", "0"))
PLS_COMP = int(os.environ.get("ATTESTRA_PLS_COMP", "32"))
OUT_NAME = os.environ.get("ATTESTRA_OUT", "BENCHMARK_FUSION_RESULT.json")
CHAMPION = "dinov2_vitl14"   # the frozen champion the authored featurizer must beat


def _l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def _fuse_pls_correct(emb_dino, emb_clip, y, tr, val, test, seed=0):
    """LEARNED featurizer: per-block L2-norm, concat, then a supervised PLS projection FIT ON TRAIN ROWS ONLY
    (sees train labels, never val/test) -> the projected representation goes through the SAME tuned-GBM head.
    Strict select-then-bound: PLS.fit uses only tr rows; transform is applied to all rows; the GBM does its own
    train/val/sealed split on the same ids; the sealed rows never influence the projection or the head fit."""
    from sklearn.cross_decomposition import PLSRegression
    X = np.concatenate([_l2(emb_dino), _l2(emb_clip)], axis=1)
    k = int(min(PLS_COMP, X.shape[1], len(tr) - 1))
    pls = PLSRegression(n_components=max(2, k), scale=False)
    pls.fit(X[list(tr)], y[list(tr)].astype(float))   # TRAIN ONLY -- no val/test leakage
    Xp = pls.transform(X).astype(np.float32)
    return BB._emb_strong_correct(Xp, y, tr, val, test, seed)


def _frozen_hashes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()[:8]
            for f in ("vectorforge/science.py", "vfplatform/sealed.py")}


COMPARISONS = [
    ("p_concat_gt_dino", "lift_concat_vs_dino", "FUSE-CONCAT vs DINOv2-L(champion)"),
    ("p_pls_gt_dino", "lift_pls_vs_dino", "FUSE-PLS vs DINOv2-L(champion)"),
    ("p_concat_gt_clip", "lift_concat_vs_clip", "FUSE-CONCAT vs CLIP-B"),
    ("p_dino_gt_clip", "lift_dino_vs_clip", "DINOv2-L(champion) vs CLIP-B (sanity, reproduces #3)"),
]


def _summarize(rows):
    summary = {}
    for key, liftk, label in COMPARISONS:
        ok = [r for r in rows if key in r]
        ps = [r[key] for r in ok]
        rej = set(benjamini_hochberg(ps, alpha=0.1)) if ps else set()
        wins = [ok[i]["name"] for i in range(len(ok)) if i in rej and ok[i][liftk] > 0]
        ml = round(float(np.mean([r[liftk] for r in ok])), 4) if ok else None
        summary[label] = {"survivors": wins, "n": len(ok), "mean_lift": ml}
    return summary


def _write_out(rows, dst):
    out = {"arena": "fgvc-aircraft", "per_class": PER_CLASS, "champion": CHAMPION, "pls_comp": PLS_COMP,
           "rows": rows, "summary": _summarize(rows), "frozen_hashes": _frozen_hashes()}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    suite = A.SUITE[:SMOKE] if SMOKE else A.SUITE
    print(f"[#4 fusion] pairs={len(suite)} per_class={PER_CLASS} champion={CHAMPION} pls_comp={PLS_COMP} "
          f"out={OUT_NAME}")
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", OUT_NAME)

    # Phase A: CLIP-B resident -> identical sealed split per pair + CLIP-B emb-strong; keep CLIP embeddings.
    cache = {}
    clip_bb = A.FROZEN["clip_vitb32"]()
    print(f"[A] clip_vitb32 dim={clip_bb.feat_dim} resident")
    for a, b in suite:
        name = f"{a}_vs_{b}"
        paths, y = A._task_paths(a, b, PER_CLASS, 0)
        emb_clip = A._embed(clip_bb, A._open_rgb(paths))
        tr, val, test = BB._sealed_split(emb_clip, y, 0)
        c, acc = BB._emb_strong_correct(emb_clip, y, tr, val, test, 0)
        cache[name] = {"y": y, "tr": tr, "val": val, "test": test, "emb_clip": emb_clip,
                       "clip_c": c, "clip_acc": round(acc, 4), "clip_lb": B1._lb(c)}
        print(f"  [A] {name:20} clip={acc:.3f}(lb{B1._lb(c)}) n_test={len(test)}")
    del clip_bb; gc.collect()

    # Phase B: DINOv2-L resident -> champion emb-strong on identical rows; keep DINOv2 embeddings.
    dino_bb = A.FROZEN["dinov2_vitl14"]()
    print(f"[B] dinov2_vitl14 dim={dino_bb.feat_dim} resident")
    for a, b in suite:
        name = f"{a}_vs_{b}"; ci = cache[name]; t0 = time.time()
        emb_dino = A._embed(dino_bb, A._open_rgb(A._task_paths(a, b, PER_CLASS, 0)[0]))
        c, acc = BB._emb_strong_correct(emb_dino, ci["y"], ci["tr"], ci["val"], ci["test"], 0)
        ci["emb_dino"] = emb_dino; ci["dino_c"] = c; ci["dino_acc"] = round(acc, 4); ci["dino_lb"] = B1._lb(c)
        print(f"  [B] {name:20} dino={acc:.3f}(lb{B1._lb(c)}) [{time.time()-t0:.0f}s]")
    del dino_bb; gc.collect()

    # Phase C: authored FEATURIZERS (no backbone resident) -> fuse/transform frozen reps, SAME tuned-GBM head.
    print("[C] authored featurizers (fuse_concat, fuse_pls) vs champion")
    rows = []
    for a, b in suite:
        name = f"{a}_vs_{b}"; ci = cache[name]; t0 = time.time()
        Xcat = np.concatenate([ci["emb_dino"], ci["emb_clip"]], axis=1).astype(np.float32)
        cc, acc_c = BB._emb_strong_correct(Xcat, ci["y"], ci["tr"], ci["val"], ci["test"], 0)
        pc, acc_p = _fuse_pls_correct(ci["emb_dino"], ci["emb_clip"], ci["y"], ci["tr"], ci["val"], ci["test"], 0)
        r = {
            "name": name, "n_test": len(ci["test"]),
            "acc": {"clip_vitb32": ci["clip_acc"], "dinov2_vitl14": ci["dino_acc"],
                    "fuse_concat": round(acc_c, 4), "fuse_pls": round(acc_p, 4)},
            "lb": {"clip_vitb32": ci["clip_lb"], "dinov2_vitl14": ci["dino_lb"],
                   "fuse_concat": B1._lb(cc), "fuse_pls": B1._lb(pc)},
            "lift_concat_vs_dino": round(acc_c - ci["dino_acc"], 4),
            "lift_pls_vs_dino": round(acc_p - ci["dino_acc"], 4),
            "lift_concat_vs_clip": round(acc_c - ci["clip_acc"], 4),
            "lift_dino_vs_clip": round(ci["dino_acc"] - ci["clip_acc"], 4),
            "p_concat_gt_dino": mcnemar_pvalue(cc, ci["dino_c"]),
            "p_pls_gt_dino": mcnemar_pvalue(pc, ci["dino_c"]),
            "p_concat_gt_clip": mcnemar_pvalue(cc, ci["clip_c"]),
            "p_dino_gt_clip": mcnemar_pvalue(ci["dino_c"], ci["clip_c"]),
            "secs": round(time.time() - t0, 1),
        }
        rows.append(r)
        print(f"  [C] {name:20} clip={ci['clip_acc']:.3f} dino={ci['dino_acc']:.3f} "
              f"concat={acc_c:.3f} pls={acc_p:.3f} | concat>dino {r['lift_concat_vs_dino']:+.3f}"
              f"(p{r['p_concat_gt_dino']:.2f}) pls>dino {r['lift_pls_vs_dino']:+.3f}(p{r['p_pls_gt_dino']:.2f})")
        ci.pop("emb_clip", None); ci.pop("emb_dino", None); gc.collect()
        _write_out(rows, dst)

    out = _write_out(rows, dst)
    print("\n=== BH-FDR(0.1) survivors with positive lift:")
    for _, _, label in COMPARISONS:
        s = out["summary"][label]
        print(f"  {label:52} {len(s['survivors'])}/{s['n']} -> {s['survivors']}  (mean lift {s['mean_lift']:+})")
    print(f"\nwrote {dst}\nfrozen: {out['frozen_hashes']}")


if __name__ == "__main__":
    main()
