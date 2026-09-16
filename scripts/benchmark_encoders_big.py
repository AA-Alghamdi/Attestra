"""Phase #5 -- DOES A BIGGER/STRONGER FROZEN ENCODER BEAT THE CHAMPION? The five prior phases established one
law across two arenas: given a representation, nothing AUTHORED on top moves the metric -- not the search cycle
(B1 0/5), not LLM-authored classifiers (B2 0/5), not end-to-end fine-tuning (#3 0/10 vs the best frozen rep),
not an authored featurizer (#4 0/10 concat, 0/10 PLS). The ONLY lever that ever moved it was CHANGING the
representation (CLIP/DINOv2 over resnet18: 5/5; DINOv2-L over CLIP-B on the hard FGVC arena: 2/10). So the
pre-registered frontier test is: does an even STRONGER frozen encoder beat DINOv2-L on the same hard arena?

  - dinov2_g    : DINOv2 ViT-g/14 (1.1B, 1536-d)        -> scale the WINNING self-supervised family L -> g
  - eva02_l     : EVA-02 ViT-L/14 MIM (304M, 1024-d)    -> a DIFFERENT strong SSL (masked-image) at L scale
  - siglip_so   : SigLIP SO400M/14 (428M, 1152-d)       -> the strongest LANGUAGE-aligned encoder at scale

If a challenger FDR-survives over DINOv2-L, the frontier lever is "find/curate the best representation" -- a
real, certifiable capability. If even the largest encoders PLATEAU here (as CLIP-L did on B1), the residual gap
on this hard low-shot arena is a DATA ceiling (label budget / class count), which reframes the mission.

DISCIPLINE is byte-identical to #3/#4: the train/val/test row-ids are derived ONCE from the CLIP-B baseline
embeddings (`benchmark_backbones._sealed_split`) and reused verbatim, so all McNemar pairing is on identical
sealed rows; every arm uses the SAME strong tuned-GBM head (`benchmark_backbones._emb_strong_correct`) so ONLY
the encoder differs; quality is the frozen Clopper-Pearson lower bound (read-only). The DINOv2-L vs CLIP-B column
MUST reproduce #3 exactly (2/10, +0.103) -- a built-in proof that the sealed split is identical. Comparisons:
one-sided exact McNemar on identical sealed rows for (challenger > dinov2), then BH-FDR(0.1) across the 10-pair
suite, per challenger. Embeddings are cached to disk per (encoder,pair) so the multi-hour run is resumable and a
per-encoder failure (e.g. OOM on the 1.1B model) is isolated -- the other encoders still report.
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
from vfplatform.featurizers import TimmBackbone                    # noqa: E402
import scripts.benchmark_vision_transfer as B1                     # noqa: E402
import scripts.benchmark_backbones as BB                           # noqa: E402
import scripts.benchmark_aircraft as A                             # noqa: E402

PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "100"))
SMOKE = int(os.environ.get("ATTESTRA_SMOKE", "0"))
OUT_NAME = os.environ.get("ATTESTRA_OUT", "BENCHMARK_ENCODERS_BIG_RESULT.json")
CACHE_DIR = os.environ.get("ATTESTRA_EMB_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "_emb_cache_phase5"))
CHAMPION = "dinov2_vitl14"   # the frozen champion every challenger must beat

# Bigger/stronger frozen encoders, all at 224px (interpolated pos-embed) so they reuse the proven DINOv2-L path
# and stay CPU-feasible. Small batches keep peak RSS bounded (the 1.1B dinov2_g is ~4.5GB fp32 on a 7GB box).
CHALLENGERS = [
    ("siglip_so", "SigLIP SO400M/14 (428M)",
     lambda: TimmBackbone(model_name="vit_so400m_patch14_siglip_224.webli", pretrained=True,
                          batch_size=16, img_size=224)),
    ("eva02_l", "EVA-02 ViT-L/14 MIM (304M)",
     lambda: TimmBackbone(model_name="eva02_large_patch14_224.mim_m38m", pretrained=True,
                          batch_size=16, img_size=224)),
    ("dinov2_g", "DINOv2 ViT-g/14 (1.1B)",
     lambda: TimmBackbone(model_name="vit_giant_patch14_dinov2.lvd142m", pretrained=True,
                          batch_size=4, img_size=224)),
]


def _frozen_hashes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()[:8]
            for f in ("vectorforge/science.py", "vfplatform/sealed.py")}


def _emb_cached(tag, name, backbone, paths):
    """Encode `paths` with `backbone`, caching the (encoder,pair) embedding matrix to disk so the multi-hour
    run is resumable and a rerun is free. The cache key includes per_class + n so a config change misses."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fp = os.path.join(CACHE_DIR, f"{tag}__{name}__pc{PER_CLASS}_n{len(paths)}.npy")
    if os.path.exists(fp):
        return np.load(fp)
    emb = A._embed(backbone, A._open_rgb(paths)).astype(np.float32)
    np.save(fp, emb)
    return emb


def _summarize(rows, tags):
    summary = {}
    # each challenger vs the champion, FDR-controlled across the 10-pair suite
    for tag, label in tags:
        liftk, pk = f"lift_{tag}_vs_dino", f"p_{tag}_gt_dino"
        ok = [r for r in rows if pk in r]
        ps = [r[pk] for r in ok]
        rej = set(benjamini_hochberg(ps, alpha=0.1)) if ps else set()
        wins = [ok[i]["name"] for i in range(len(ok)) if i in rej and ok[i][liftk] > 0]
        ml = round(float(np.mean([r[liftk] for r in ok])), 4) if ok else None
        summary[f"{label} vs DINOv2-L(champion)"] = {"survivors": wins, "n": len(ok), "mean_lift": ml}
    # sanity: DINOv2-L vs CLIP-B must reproduce #3 (2/10, +0.103) -> identical sealed split
    ok = [r for r in rows if "p_dino_gt_clip" in r]
    ps = [r["p_dino_gt_clip"] for r in ok]
    rej = set(benjamini_hochberg(ps, alpha=0.1)) if ps else set()
    wins = [ok[i]["name"] for i in range(len(ok)) if i in rej and ok[i]["lift_dino_vs_clip"] > 0]
    summary["DINOv2-L(champion) vs CLIP-B (sanity, reproduces #3)"] = {
        "survivors": wins, "n": len(ok),
        "mean_lift": round(float(np.mean([r["lift_dino_vs_clip"] for r in ok])), 4) if ok else None}
    return summary


def _write_out(rows, tags, dst):
    out = {"arena": "fgvc-aircraft", "per_class": PER_CLASS, "champion": CHAMPION,
           "challengers": [t for t, _ in tags], "rows": rows,
           "summary": _summarize(rows, tags), "frozen_hashes": _frozen_hashes()}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    suite = A.SUITE[:SMOKE] if SMOKE else A.SUITE
    challengers = CHALLENGERS[:int(os.environ.get("ATTESTRA_N_CHAL", len(CHALLENGERS)))]
    print(f"[#5 encoders] pairs={len(suite)} per_class={PER_CLASS} champion={CHAMPION} "
          f"challengers={[t for t, _, _ in challengers]} out={OUT_NAME}")
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", OUT_NAME)

    # Phase A: CLIP-B resident -> identical sealed split per pair (same as #3/#4) + CLIP-B emb-strong baseline.
    cache = {}
    clip_bb = A.FROZEN["clip_vitb32"]()
    print(f"[A] clip_vitb32 dim={clip_bb.feat_dim} resident")
    for a, b in suite:
        name = f"{a}_vs_{b}"
        paths, y = A._task_paths(a, b, PER_CLASS, 0)
        emb_clip = _emb_cached("clip_vitb32", name, clip_bb, paths)
        tr, val, test = BB._sealed_split(emb_clip, y, 0)
        c, acc = BB._emb_strong_correct(emb_clip, y, tr, val, test, 0)
        cache[name] = {"paths": paths, "y": y, "tr": tr, "val": val, "test": test,
                       "clip_c": c, "clip_acc": round(acc, 4), "clip_lb": B1._lb(c)}
        print(f"  [A] {name:20} clip={acc:.3f}(lb{B1._lb(c)}) n_test={len(test)}")
    del clip_bb; gc.collect()

    # Phase B: DINOv2-L resident -> champion emb-strong on identical rows (MUST reproduce #3/#4).
    dino_bb = A.FROZEN["dinov2_vitl14"]()
    print(f"[B] dinov2_vitl14 dim={dino_bb.feat_dim} resident")
    for a, b in suite:
        name = f"{a}_vs_{b}"; ci = cache[name]; t0 = time.time()
        emb_dino = _emb_cached("dinov2_vitl14", name, dino_bb, ci["paths"])
        c, acc = BB._emb_strong_correct(emb_dino, ci["y"], ci["tr"], ci["val"], ci["test"], 0)
        ci["dino_c"] = c; ci["dino_acc"] = round(acc, 4); ci["dino_lb"] = B1._lb(c)
        print(f"  [B] {name:20} dino={acc:.3f}(lb{B1._lb(c)}) [{time.time()-t0:.0f}s]")
    del dino_bb; gc.collect()

    # rows seeded with the fixed baselines; each challenger fills in its own columns as it runs.
    rows = []
    for a, b in suite:
        name = f"{a}_vs_{b}"; ci = cache[name]
        rows.append({
            "name": name, "n_test": len(ci["test"]),
            "acc": {"clip_vitb32": ci["clip_acc"], "dinov2_vitl14": ci["dino_acc"]},
            "lb": {"clip_vitb32": ci["clip_lb"], "dinov2_vitl14": ci["dino_lb"]},
            "lift_dino_vs_clip": round(ci["dino_acc"] - ci["clip_acc"], 4),
            "p_dino_gt_clip": mcnemar_pvalue(ci["dino_c"], ci["clip_c"]),
        })
    row_by = {r["name"]: r for r in rows}
    tags = []
    _write_out(rows, tags, dst)

    # Phase C: each BIGGER encoder resident ONE AT A TIME -> encode identical rows, emb-strong, McNemar vs champion.
    for tag, label, factory in challengers:
        print(f"[C] {tag} ({label}) loading...")
        try:
            bb = factory()
        except Exception as exc:  # noqa: BLE001  -- isolate a per-encoder load failure (e.g. OOM/download)
            print(f"  [C] {tag} LOAD FAILED: {type(exc).__name__}: {exc} -- skipping, other encoders continue")
            continue
        print(f"  [C] {tag} dim={bb.feat_dim} resident")
        try:
            for a, b in suite:
                name = f"{a}_vs_{b}"; ci = cache[name]; r = row_by[name]; t0 = time.time()
                emb = _emb_cached(tag, name, bb, ci["paths"])
                cc, acc = BB._emb_strong_correct(emb, ci["y"], ci["tr"], ci["val"], ci["test"], 0)
                r["acc"][tag] = round(acc, 4)
                r["lb"][tag] = B1._lb(cc)
                r[f"lift_{tag}_vs_dino"] = round(acc - ci["dino_acc"], 4)
                r[f"lift_{tag}_vs_clip"] = round(acc - ci["clip_acc"], 4)
                r[f"p_{tag}_gt_dino"] = mcnemar_pvalue(cc, ci["dino_c"])
                r[f"p_{tag}_gt_clip"] = mcnemar_pvalue(cc, ci["clip_c"])
                r[f"secs_{tag}"] = round(time.time() - t0, 1)
                del emb; gc.collect()
                print(f"  [C] {tag:10} {name:20} acc={acc:.3f}(lb{B1._lb(cc)}) "
                      f"vs dino {r[f'lift_{tag}_vs_dino']:+.3f}(p{r[f'p_{tag}_gt_dino']:.2f}) [{time.time()-t0:.0f}s]")
                tags_now = tags + [(tag, label)] if (tag, label) not in tags else tags
                _write_out(rows, tags_now, dst)   # checkpoint after every pair
        finally:
            del bb; gc.collect()
        if (tag, label) not in tags:
            tags.append((tag, label))
        _write_out(rows, tags, dst)

    out = _write_out(rows, tags, dst)
    print("\n=== BH-FDR(0.1) survivors with positive lift:")
    for label, s in out["summary"].items():
        print(f"  {label:52} {len(s['survivors'])}/{s['n']} -> {s['survivors']}  (mean lift {s['mean_lift']:+})")
    print(f"\nwrote {dst}\nfrozen: {out['frozen_hashes']}")


if __name__ == "__main__":
    main()
