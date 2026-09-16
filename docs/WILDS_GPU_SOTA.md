# Phase 1 (GPU): Regenerative recipe discovery vs a human-grade ViT-L baseline

**The absolute-SOTA arena.** Phase 0 (`scripts/wilds_arena.py`) proved the *machinery* on real WILDS
distribution shift by realising each recipe as a frozen-feature probe (the backbone is the only live lever).
Phase 1 honours the **full recipe** on a GPU and asks the absolute question the user locked:

> Does a *menu-free, regenerated* recipe **beat a mid-level engineer's ViT-L fine-tune** on the identical
> sealed rows, under the frozen certifier?

The spine is unchanged: **the generator regenerates full recipes → the GPU runner executes them → the
frozen certifier promotes on sealed evidence.** Nothing new is allowed to mint a certificate.

## One command

```bash
# GPU day (the real run):
ATTESTRA_DEVICE=cuda python scripts/run_wilds_gpu.py

# CPU pipeline smoke (tiny, no GPU; proves the plumbing end-to-end):
ATTESTRA_DEVICE=cpu python scripts/run_wilds_gpu.py --smoke
```

Emits `docs/WILDS_GPU_SOTA_CERTIFICATE.json`: the discovery certificate (menu-free champion) **plus** the
head-to-head absolute-SOTA verdict.

## What the GPU runner actually executes (the recipe is honoured, not recorded)

`scripts/wilds_gpu.py::GpuWildsArena` realises every recipe axis on the real images:

| Recipe axis | How Phase 1 honours it |
|---|---|
| `adaptation` | `linear_probe` (freeze backbone, train head) · `lora`/`adapter` (PEFT deltas on attn/MLP linears + head) · `vpt` (VPT-shallow learnable prompt tokens on a ViT) · `partial_unfreeze` (top block + head) · `full_ft` (all params) |
| `augmentation` | `randaug` / `trivialaug` (torchvision) · `mixup` / `cutmix` (timm `Mixup`) · flip baseline |
| `optimizer` | `adamw` · `sam` (Sharpness-Aware Minimisation two-step) |
| `schedule` | `cosine` · `step` · `constant` (+ `llrd` recorded) |
| `aggregation` | `single` · `logit_ensemble` (mean softmax of N members) · `model_soup` (uniform weight-space average, Wortsman 2022) |
| `data_strategy` | `class_rebalance` (weighted sampler) |
| `epochs`, `lr`, `weight_decay` | passed straight into the optimiser/loop |

Device-agnostic: `autocast`/`GradScaler` engage only on CUDA; the same code runs on CPU for the smoke.
Caps `ATTESTRA_GPU_MAX_TRAIN` / `ATTESTRA_GPU_EPOCH_CAP` / `ATTESTRA_GPU_BATCH` shrink the work for the smoke
and are unset on the real run (the full recipe is honoured).

## The absolute-SOTA referee (`_certify_head_to_head`)

Both the discovered champion and `human_baseline_recipe()` (a strong **ViT-L full fine-tune** with RandAugment
+ cosine + layer-wise LR decay - the conventional mid-level default) are scored on the **identical carved
sealed shards**. Then:

- **paired McNemar** per shard, **BH-FDR(α)** across shards → does the champion *beat* the human?
- **frozen Clopper–Pearson** lower bounds on both pooled sealed accuracies.
- a **never-peeked gold** read on the disjoint center-2 gold set.

```
absolute_sota_earned  ==  (FDR survivors with positive lift) AND (champion_sealed_lb > human_sealed_acc)
                          AND (gold: champion_lb > human_acc AND champion_acc > human_acc)
```

The select-then-bound discipline is inherited verbatim from `Camelyon17Splits` (sealed shards carved once,
reused for every recipe and for the human baseline). The certifier is the sole arbiter; this function only
assembles its primitives over identical rows.

## Status / verification before GPU day

- `tests/test_wilds_gpu.py` - 13 hermetic locks (offline, `pretrained=False`, CPU): adaptations wire
  (linear_probe freezes, full_ft trains all, partial unfreezes a strict subset, LoRA injects deltas, VPT adds
  a trainable prompt) · select-then-bound identical sealed rows · model-soup collapses members · the
  absolute-SOTA referee earns the claim only when the champion truly beats the human, and refuses on a tie.
- `tests/test_wilds_arena.py` - 4 locks on the Phase-0 carving + framing.
- Frozen certifier unchanged: `vectorforge/science.py b564fba2`, `vfplatform/sealed.py 30ad6245`.

GPU day is then exactly one command, with no download wait for the seed backbones (prefetched, see
`docs/GPU_READINESS.md`).
