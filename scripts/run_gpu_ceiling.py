"""#5 -- THE GPU-DAY EXPERIMENT, PRE-WIRED: does an end-to-end fine-tune beat the best FROZEN encoder on ABSOLUTES?

This is the one command tomorrow turns into the "beats a mid-level engineer on absolutes" test the audit
flagged. It fine-tunes a real backbone (default `vit_l_16` SWAG on GPU -- what a human would actually reach for)
end-to-end on the IDENTICAL FGVC-Aircraft sealed rows the autonomous champion was certified on, and certifies the
fine-tune vs the frozen champion (DINOv2-g) under the EXACT same discipline as every prior phase: paired one-sided
McNemar on the identical sealed rows, Benjamini-Hochberg(0.1) across the suite, frozen Clopper-Pearson bound.

It reuses the unified `FgvcAircraftArena` so the split, the frozen-champion correctness, and the row ordering are
byte-identical to the certified run; the fine-tune arm is the device-agnostic `gpu_finetune.finetune_correct`, so
the ONLY thing that changes tomorrow is `ATTESTRA_DEVICE=cuda`. Tonight it is validated on CPU with a 1-pair smoke
(`ATTESTRA_SMOKE=1`, tiny epochs); the numbers there are throwaway, the WIRING is the point.

  GPU day:  ATTESTRA_DEVICE=cuda ATTESTRA_GPU_FT_ARCH=vit_l_16 python scripts/run_gpu_ceiling.py
  tonight:  ATTESTRA_SMOKE=1 ATTESTRA_GPU_FT_ARCH=resnet18 ATTESTRA_GPU_FT_EPOCHS=2 python scripts/run_gpu_ceiling.py
"""
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue          # noqa: E402
import scripts.benchmark_aircraft as A                                     # noqa: E402
import scripts.benchmark_vision_transfer as B1                            # noqa: E402
import scripts.gpu_finetune as G                                          # noqa: E402
from scripts.repr_arena import FgvcAircraftArena                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
CHAMP_TAG = os.environ.get("ATTESTRA_CHAMPION", "dinov2_g")    # the frozen champion the fine-tune must beat
FT_ARCH = os.environ.get("ATTESTRA_GPU_FT_ARCH", "vit_l_16")
FT_PX = int(os.environ.get("ATTESTRA_GPU_FT_PX", "224"))
FT_EPOCHS = int(os.environ.get("ATTESTRA_GPU_FT_EPOCHS", "40"))
FT_LR = float(os.environ.get("ATTESTRA_GPU_FT_LR", "0.003"))
FT_RESTARTS = int(os.environ.get("ATTESTRA_GPU_FT_RESTARTS", "3"))
SMOKE = int(os.environ.get("ATTESTRA_SMOKE", "0"))
ALPHA = 0.1


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


def _aligned_champion_correct(arena, task, order):
    """The frozen champion's sealed correctness on `task`, reordered to `order` (sorted test-row order) so it
    pairs row-for-row with the fine-tune's sealed correctness in McNemar. The arena scores rows in sp['test']
    order; `order` is the argsort that maps that to sorted(test_ids)."""
    champ = arena.measure(CHAMP_TAG)[task].sealed_correct
    return [int(champ[j]) for j in order]


def run():
    hashes = _frozen_hashes()
    assert hashes == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {hashes} != {FROZEN_EXPECTED}"
    device = G.pick_device()
    arena = FgvcAircraftArena(smoke=SMOKE)
    print(f"[GPU-ceiling] device={device} champion={CHAMP_TAG} ft={FT_ARCH}@{FT_PX}px x{FT_EPOCHS}ep "
          f"lr{FT_LR} restarts={FT_RESTARTS} tasks={len(arena.tasks)}")
    print(f"frozen certifier verified: {hashes}")

    rows = []
    for task in arena.tasks:
        a, b = arena._pairs[task]
        sp = arena._task_split(task)
        y = np.asarray(sp["y"])
        tr, val, test = np.asarray(sp["tr"]), np.asarray(sp["val"]), np.asarray(sp["test"])
        order = np.argsort(test)                       # arena scores in sp['test'] order; FT scores sorted
        sorted_test = test[order]
        champ_c = _aligned_champion_correct(arena, task, order)
        champ_acc = float(np.mean(champ_c))

        paths, _ = A._task_paths(a, b, arena.per_class, 0)
        dc = A._decode_cache(paths, FT_PX)
        t0 = time.time()
        ft_c, ft_acc, best_ep, best_va, vals = G.finetune_correct(
            dc, y, tr, val, sorted_test, arch=FT_ARCH, px=FT_PX, epochs=FT_EPOCHS, lr=FT_LR,
            restarts=FT_RESTARTS, device=device)
        del dc; gc.collect()

        assert len(ft_c) == len(champ_c) == len(test), "sealed row-count mismatch between FT and champion"
        p_ft_gt_champ = mcnemar_pvalue(list(ft_c), list(champ_c))
        rows.append({
            "task": task, "n_test": int(len(test)),
            "champion": CHAMP_TAG, "champion_acc": round(champ_acc, 4), "champion_lb": B1._lb(champ_c),
            "ft_acc": round(ft_acc, 4), "ft_lb": B1._lb(list(ft_c)),
            "ft_best_ep": best_ep, "ft_val": best_va, "ft_restart_vals": vals,
            "lift_ft_vs_champion": round(ft_acc - champ_acc, 4),
            "p_ft_gt_champion": p_ft_gt_champ, "secs": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"  {task:22} champ={champ_acc:.3f}(lb{r['champion_lb']}) ft={ft_acc:.3f}(lb{r['ft_lb']}) "
              f"lift{r['lift_ft_vs_champion']:+.3f}(p{p_ft_gt_champ:.3f}) ep{best_ep} [{r['secs']}s]")

    ps = [r["p_ft_gt_champion"] for r in rows]
    rej = set(benjamini_hochberg(ps, alpha=ALPHA)) if ps else set()
    survivors = [rows[i]["task"] for i in range(len(rows)) if i in rej and rows[i]["lift_ft_vs_champion"] > 0]
    mean_lift = round(float(np.mean([r["lift_ft_vs_champion"] for r in rows])), 4) if rows else None

    out = {
        "arena": "fgvc-aircraft", "device": str(device), "champion": CHAMP_TAG,
        "ft_arch": FT_ARCH, "ft_px": FT_PX, "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR, "ft_restarts": FT_RESTARTS,
        "smoke": SMOKE, "rows": rows,
        "summary": {"comparison": f"FINE-TUNE({FT_ARCH}) vs FROZEN-CHAMPION({CHAMP_TAG})",
                    "fdr_survivors": survivors, "n": len(rows), "mean_lift": mean_lift,
                    "fine_tune_beats_frozen_champion_on_absolutes": len(survivors) > 0},
        "frozen_hashes": _frozen_hashes(),
    }
    name = "GPU_CEILING_SMOKE.json" if SMOKE else "GPU_CEILING_RESULT.json"
    dst = os.path.join(ROOT, "docs", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== BH-FDR({ALPHA}) FINE-TUNE({FT_ARCH}) vs FROZEN-CHAMPION({CHAMP_TAG}): "
          f"{len(survivors)}/{len(rows)} survivors, mean lift {mean_lift:+} -> {survivors}")
    print(f"wrote {dst}\nfrozen (post-run): {_frozen_hashes()}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
