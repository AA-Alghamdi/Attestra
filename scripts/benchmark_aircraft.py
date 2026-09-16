"""Phase #3 -- FGVC-AIRCRAFT FINE-GRAINED ARENA: how much headroom do the BEST FROZEN features leave, and
does an end-to-end FINE-TUNE ceiling capture it?

B1/B2/B2-repr/B2-repr-2 established a clean law on confusable CIFAR-10: given a representation, neither the
search cycle (0/5) nor LLM-authored novel methods (0/5) beat a tuned GBM; the representation is the lever, and
the best frozen backbones (CLIP-B/DINOv2) drive that arena to a ~0.99 ceiling -- there is no FDR-measurable
room left. To keep "beats a competent mid-level engineer" FALSIFIABLE we need a harder arena where even the
best frozen features leave a gap a human closes by FINE-TUNING. FGVC-Aircraft variant pairs are exactly that:
frozen CLIP-B scores only ~0.58-0.67 on confusable same-family pairs (737-700 vs 737-800, A330-200 vs -300,
ERJ 135 vs 145) -- ~30 points below the ceiling.

The arena is a suite of maximally-confusable SAME-FAMILY adjacent variant binary tasks (same machinery as B1:
per task an IDENTICAL sealed test held out before any fit; the strong head is a tuned GBM + random search on
the catalog; the frozen Clopper-Pearson lower bound is read-only `science.clopper_pearson_lower`). Three arms,
each scored on the IDENTICAL sealed rows so every McNemar pairing is exact:

  clip_vitb32   : the deployed FROZEN champion (CLIP ViT-B/32) + the strong tuned-GBM head   -> the BASELINE
  dinov2_vitl14 : the strongest FROZEN family (DINOv2 ViT-L/14) + the same strong head        -> frozen rival
  finetune      : an end-to-end FINE-TUNED ImageNet CNN (resnet50, partial unfreeze) on the SAME train rows,
                  model-selected on the SAME val rows -> the CEILING that adapts the representation itself

Paired one-sided exact McNemar on the sealed rows for (dinov2 > clip), (finetune > clip), (finetune > dinov2),
then Benjamini-Hochberg(0.1) across the suite per comparison. The headline is the frozen->fine-tune GAP: if the
fine-tune ceiling FDR-survives over the best frozen rep, this arena has real headroom above frozen features
(the place the frontier-vs-human claim stays measurable); if it does not, even a CPU fine-tune cannot capture
it and the lever lies elsewhere. The frozen certifier is the sole promoter and is never edited; this script
only MEASURES and reports honest results (positive or negative) as first-class.
"""
import copy
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorforge import science                                    # noqa: E402
from vfplatform.battery import mcnemar_pvalue, benjamini_hochberg  # noqa: E402
from vfplatform.featurizers import (ImageFeaturizer, ClipImageBackbone,  # noqa: E402
                                    TimmBackbone)
import scripts.benchmark_vision_transfer as B1                     # noqa: E402
import scripts.benchmark_backbones as BB                           # noqa: E402

DATA_DIR = os.environ.get("ATTESTRA_DATA_AIR", os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "data"))
PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "100"))        # FGVC has exactly 100 imgs/variant
ALPHA = 0.05
FT_ARCH = os.environ.get("ATTESTRA_FT_ARCH", "resnet50")
FT_EPOCHS = int(os.environ.get("ATTESTRA_FT_EPOCHS", "30"))
FT_PX = int(os.environ.get("ATTESTRA_FT_PX", "224"))
FT_LR = float(os.environ.get("ATTESTRA_FT_LR", "0.005"))
FT_RESTARTS = int(os.environ.get("ATTESTRA_FT_RESTARTS", "3"))     # val-selected restarts (no test peeking)
SMOKE = int(os.environ.get("ATTESTRA_SMOKE", "0"))                  # >0 -> run only first SMOKE pairs
OUT_NAME = os.environ.get("ATTESTRA_OUT", "BENCHMARK_AIRCRAFT_RESULT.json")

# Maximally-confusable SAME-FAMILY adjacent variant pairs -> genuine headroom above the best frozen features.
SUITE = [
    ("737-700", "737-800"),     # 737 NG adjacent stretch
    ("737-300", "737-400"),     # 737 Classic adjacent stretch
    ("747-200", "747-400"),     # 747 jumbo generations
    ("A330-200", "A330-300"),   # A330 length variants
    ("A340-300", "A340-600"),   # A340 short vs long
    ("CRJ-700", "CRJ-900"),     # regional-jet stretch
    ("ERJ 135", "ERJ 145"),     # Embraer regional twins
    ("E-190", "E-195"),         # E-Jet adjacent stretch
    ("767-300", "767-400"),     # 767 stretch
    ("DHC-8-100", "DHC-8-300"),  # Dash-8 length variants
]

FROZEN = {
    "clip_vitb32": lambda: ClipImageBackbone(model_name="ViT-B-32-quickgelu", pretrained="openai",
                                             batch_size=64),
    "dinov2_vitl14": lambda: TimmBackbone(model_name="vit_large_patch14_dinov2.lvd142m",
                                          pretrained=True, batch_size=8, img_size=224),
}
FROZEN_BASELINE = "clip_vitb32"   # the deployed champion we bound the fine-tune gap against

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- data
_POOL = None


def _pool():
    """variant -> list of image file paths, pooling FGVC trainval+test (100 imgs/variant total)."""
    global _POOL
    if _POOL is not None:
        return _POOL
    import torchvision.datasets as DS
    pool = {}
    for split in ("trainval", "test"):
        ds = DS.FGVCAircraft(root=DATA_DIR, split=split, annotation_level="variant", download=False)
        for path, lab in zip(ds._image_files, ds._labels):
            pool.setdefault(ds.classes[lab], []).append(str(path))
    _POOL = pool
    return pool


def _task_paths(cls_a, cls_b, per_class=PER_CLASS, seed=0):
    """Return (list of image paths, y) for a confusable pair -- same shuffle discipline as B1._task_data."""
    pool = _pool()
    rng = np.random.RandomState(seed)
    out_paths, out_y = [], []
    for cls, y in [(cls_a, 0), (cls_b, 1)]:
        paths = list(pool[cls])
        rng.shuffle(paths)
        for p in paths[:per_class]:
            out_paths.append(p); out_y.append(y)
    order = rng.permutation(len(out_y))
    return [out_paths[i] for i in order], np.array([out_y[i] for i in order])


def _open_rgb(paths):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in paths]


def _embed(backbone, pil_imgs):
    return ImageFeaturizer(backbone=backbone, backbone_dim=backbone.feat_dim).transform(pil_imgs)


# ----------------------------------------------------------------------------- fine-tune ceiling arm
def _decode_cache(paths, px):
    """Pre-decode+resize each JPEG ONCE to a square (px+32) PIL so per-epoch augmentation never re-decodes the
    full-resolution file (JPEG decode dominates CPU cost). Memory ~ n*(px+32)^2*3 bytes (~40MB/task at 224)."""
    import torchvision.transforms.functional as F
    from PIL import Image
    side = px + 32
    cache = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = F.resize(im, side)            # shorter side -> side
        im = F.center_crop(im, side)       # bound to side x side
        cache.append(im)
    return cache


def _make_model(arch):
    import torch.nn as nn
    import torchvision.models as M
    if arch == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        trainable = ("layer3", "layer4", "fc")
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        trainable = ("layer3", "layer4", "fc")
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        raise SystemExit(f"unknown FT arch {arch!r}")
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(trainable)
    return m


def _set_bn_eval(model):
    """Freeze BatchNorm running stats during fine-tuning (standard small-data trick): keep pretrained stats."""
    import torch.nn as nn
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()


def _finetune_once(cache, y, tr_ids, val_ids, test_ids, arch, px, epochs, lr, seed):
    """One fine-tune restart: train end-to-end on the IDENTICAL train rows, select the best EPOCH on the
    IDENTICAL val rows, return (sealed correctness in sorted(test_ids) order, test_acc, best_val, best_ep).
    The sealed rows never influence training or epoch selection (strict select-then-bound, no leakage)."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(2)

    train_tf = T.Compose([T.RandomResizedCrop(px, scale=(0.55, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = T.Compose([T.CenterCrop(px), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class _DS(Dataset):
        def __init__(self, ids, tf):
            self.ids = list(ids); self.tf = tf

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, k):
            i = self.ids[k]
            return self.tf(cache[i]), int(y[i])

    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(_DS(tr_ids, train_tf), batch_size=16, shuffle=True, num_workers=0, generator=g)
    va = DataLoader(_DS(val_ids, eval_tf), batch_size=32, num_workers=0)
    te = DataLoader(_DS(sorted(test_ids), eval_tf), batch_size=32, num_workers=0)

    model = _make_model(arch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    def _eval(loader):
        model.eval(); corr = []
        with torch.no_grad():
            for xb, yb in loader:
                corr += (model(xb).argmax(1) == yb).int().tolist()
        return corr

    best_va, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train(); _set_bn_eval(model)
        for xb, yb in tr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        va_acc = float(np.mean(_eval(va)))
        if va_acc > best_va:
            best_va, best_state, best_ep = va_acc, copy.deepcopy(model.state_dict()), ep
        if SMOKE:
            print(f"      seed{seed} ep{ep:02d} val={va_acc:.3f} (best {best_va:.3f}@{best_ep})")
    model.load_state_dict(best_state); model.eval()
    correct = _eval(te)
    del model, best_state; gc.collect()
    return correct, float(np.mean(correct)), round(best_va, 4), best_ep


def _finetune_correct(cache, y, tr_ids, val_ids, test_ids, arch=FT_ARCH, px=FT_PX,
                      epochs=FT_EPOCHS, lr=FT_LR, restarts=FT_RESTARTS):
    """The CEILING arm as a FAIR strong attempt: run `restarts` independent fine-tune restarts (distinct seeds)
    and keep the restart with the best VAL accuracy -- standard practice that guards against the occasional
    early-divergence collapse on ~100-image classes WITHOUT ever peeking at the sealed test. Returns
    (sealed correctness, test_acc, best_ep_of_winner, best_val_of_winner, per_restart_vals)."""
    best = None
    vals = []
    for r in range(restarts):
        c, acc, va, ep = _finetune_once(cache, y, tr_ids, val_ids, test_ids, arch, px, epochs, lr, seed=r)
        vals.append(va)
        if best is None or va > best[2]:
            best = (c, acc, va, ep)
    return best[0], best[1], best[3], best[2], vals


# ----------------------------------------------------------------------------- orchestration
def _frozen_hashes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()[:8]
            for f in ("vectorforge/science.py", "vfplatform/sealed.py")}


def _summarize(rows, comparisons):
    summary = {}
    for key, liftk, label in comparisons:
        ok = [r for r in rows if key in r]
        ps = [r[key] for r in ok]
        rej = set(benjamini_hochberg(ps, alpha=0.1)) if ps else set()
        wins = [ok[i]["name"] for i in range(len(ok)) if i in rej and ok[i][liftk] > 0]
        mean_lift = round(float(np.mean([r[liftk] for r in ok])), 4) if ok else None
        summary[label] = {"survivors": wins, "n": len(ok), "mean_lift": mean_lift}
    return summary


COMPARISONS = [
    ("p_dino_gt_clip", "lift_dino_vs_clip", "DINOv2-L(frozen) vs CLIP-B(frozen)"),
    ("p_ft_gt_clip", "lift_ft_vs_clip", "FINE-TUNE(ceiling) vs CLIP-B(frozen)"),
    ("p_ft_gt_dino", "lift_ft_vs_dino", "FINE-TUNE(ceiling) vs DINOv2-L(frozen)"),
]


def _write_out(rows, dst):
    out = {"arena": "fgvc-aircraft", "per_class": PER_CLASS, "ft_arch": FT_ARCH, "ft_px": FT_PX,
           "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR, "baseline": FROZEN_BASELINE, "rows": rows,
           "summary": _summarize(rows, COMPARISONS), "frozen_hashes": _frozen_hashes()}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    suite = SUITE[:SMOKE] if SMOKE else SUITE
    print(f"[#3 aircraft] pairs={len(suite)} per_class={PER_CLASS} ft={FT_ARCH}@{FT_PX}px x{FT_EPOCHS}ep "
          f"lr{FT_LR} | baseline={FROZEN_BASELINE} | out={OUT_NAME}")
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", OUT_NAME)

    # Phase A: FROZEN baseline (CLIP-B) resident -> per task derive the IDENTICAL sealed split + baseline corr.
    cache = {}
    base_bb = FROZEN[FROZEN_BASELINE]()
    print(f"[A] frozen baseline {FROZEN_BASELINE} dim={base_bb.feat_dim} resident")
    for a, b in suite:
        name = f"{a}_vs_{b}"
        paths, y = _task_paths(a, b, PER_CLASS, 0)
        emb = _embed(base_bb, _open_rgb(paths))
        tr_ids, val_ids, test_ids = BB._sealed_split(emb, y, 0)
        c, acc = BB._emb_strong_correct(emb, y, tr_ids, val_ids, test_ids, 0)
        cache[name] = {"paths": paths, "y": y, "tr": tr_ids, "val": val_ids, "test": test_ids,
                       "clip_c": c, "clip_acc": round(acc, 4), "clip_lb": B1._lb(c)}
        print(f"  [A] {name:20} clip={acc:.3f}(lb{B1._lb(c)}) n_test={len(test_ids)}")
    del base_bb; gc.collect()

    rows = []
    for a, b in suite:
        name = f"{a}_vs_{b}"
        ci = cache[name]
        rows.append({"name": name, "n_test": len(ci["test"]), "acc": {"clip_vitb32": ci["clip_acc"]},
                     "lb": {"clip_vitb32": ci["clip_lb"]}})

    # Phase B1: DINOv2-L frozen resident -> embeddings + correctness on identical rows.
    dino_bb = FROZEN["dinov2_vitl14"]()
    print(f"[B1] frozen rival dinov2_vitl14 dim={dino_bb.feat_dim} resident")
    for r in rows:
        ci = cache[r["name"]]; t0 = time.time()
        emb = _embed(dino_bb, _open_rgb(ci["paths"]))
        c, acc = BB._emb_strong_correct(emb, ci["y"], ci["tr"], ci["val"], ci["test"], 0)
        ci["dino_c"] = c
        r["acc"]["dinov2_vitl14"] = round(acc, 4); r["lb"]["dinov2_vitl14"] = B1._lb(c)
        r["lift_dino_vs_clip"] = round(acc - ci["clip_acc"], 4)
        r["p_dino_gt_clip"] = mcnemar_pvalue(c, ci["clip_c"])
        print(f"  [B1] {r['name']:20} dino={acc:.3f}(lb{B1._lb(c)}) "
              f"lift{r['lift_dino_vs_clip']:+.3f}(p{r['p_dino_gt_clip']:.2f}) [{time.time()-t0:.0f}s]")
        del emb
    del dino_bb; gc.collect()
    _write_out(rows, dst)

    # Phase B2: FINE-TUNE ceiling -> end-to-end on identical train rows, select on val, bound on sealed test.
    print(f"[B2] fine-tune ceiling {FT_ARCH}@{FT_PX}px x{FT_EPOCHS}ep")
    for r in rows:
        ci = cache[r["name"]]; t0 = time.time()
        dc = _decode_cache(ci["paths"], FT_PX)
        c, acc, best_ep, best_va, vals = _finetune_correct(dc, ci["y"], ci["tr"], ci["val"], ci["test"])
        ci["ft_c"] = c
        r["acc"]["finetune"] = round(acc, 4); r["lb"]["finetune"] = B1._lb(c)
        r["ft_best_ep"] = best_ep; r["ft_val"] = best_va; r["ft_restart_vals"] = vals
        r["lift_ft_vs_clip"] = round(acc - ci["clip_acc"], 4)
        r["lift_ft_vs_dino"] = round(acc - r["acc"]["dinov2_vitl14"], 4)
        r["p_ft_gt_clip"] = mcnemar_pvalue(c, ci["clip_c"])
        r["p_ft_gt_dino"] = mcnemar_pvalue(c, ci["dino_c"])
        r["secs_ft"] = round(time.time() - t0, 1)
        print(f"  [B2] {r['name']:20} ft={acc:.3f}(lb{B1._lb(c)}) ep{best_ep} val{best_va} vals{vals} | "
              f"ft>clip {r['lift_ft_vs_clip']:+.3f}(p{r['p_ft_gt_clip']:.2f}) "
              f"ft>dino {r['lift_ft_vs_dino']:+.3f}(p{r['p_ft_gt_dino']:.2f}) [{r['secs_ft']}s]")
        del dc; gc.collect()
        _write_out(rows, dst)

    out = _write_out(rows, dst)
    print("\n=== BH-FDR(0.1) survivors with positive lift:")
    for _, _, label in COMPARISONS:
        s = out["summary"][label]
        print(f"  {label:38} {len(s['survivors'])}/{s['n']} -> {s['survivors']}  (mean lift {s['mean_lift']:+})")
    print(f"\nwrote {dst}\nfrozen: {out['frozen_hashes']}")


if __name__ == "__main__":
    main()
