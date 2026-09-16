# GPU-readiness (#5): tomorrow's recursion is one command

Everything the GPU box needs is built, cached, and locked tonight on this CPU-only box. GPU day is then a device
swap + one command, not a porting exercise.

## What ships

| Piece | File | What it does |
|---|---|---|
| Device-agnostic fine-tune | `scripts/gpu_finetune.py` | `pick_device()` (cuda → cpu, `$ATTESTRA_DEVICE` override), `make_model(arch)` partial-unfreeze backbones (`resnet18/50`, `vit_b_16`, `vit_l_16`), `finetune_once`/`finetune_correct` with mixed-precision autocast **on CUDA only**. Identical select-then-bound return contract as the phase-#3 ceiling arm. |
| Weight prefetch | `scripts/prefetch_gpu_weights.py` | Downloads + verifies every checkpoint tomorrow needs **tonight** (checkpoints are device-agnostic bytes). Writes `docs/GPU_WEIGHTS_MANIFEST.json`. |
| Pre-wired GPU experiment | `scripts/run_gpu_ceiling.py` | The "beats a mid-level engineer on **absolutes**" test: fine-tune (default `vit_l_16` SWAG) end-to-end on the **identical** FGVC sealed rows vs the frozen champion (DINOv2-g), certified by paired McNemar + BH-FDR(0.1) + frozen Clopper-Pearson bound. Reuses `FgvcAircraftArena`, so split / champion correctness / row order are byte-identical to the certified run. |
| Hermetic locks | `tests/test_gpu_finetune.py`, `tests/test_gpu_ceiling.py` | 8 CPU tests: device selection, **strict select-then-bound** (flipping only sealed labels leaves val-selection identical and inverts sealed correctness), determinism, arch registry, and the McNemar **row-alignment** remap. |

## Verified tonight (CPU)

- `python scripts/gpu_finetune.py --smoke` - device-agnostic fine-tune runs end-to-end on CPU.
- `python scripts/gpu_finetune.py --check-arch {resnet50,vit_b_16,vit_l_16}` - all build (ViT-L: 304M total / 50.4M trainable with last-4-block unfreeze).
- `python scripts/prefetch_gpu_weights.py` - **17.81 GB cached, ready=True**; the only download was `vit_l_16_swag` (1.14 GB), now on disk. All 13 arena encoders present in the HF cache.
- `ATTESTRA_SMOKE=1 ... python scripts/run_gpu_ceiling.py` - full wiring runs; champion reads **0.883** on 737-700/800, byte-identical to the certified DINOv2-g number → identical sealed rows confirmed.
- 8/8 hermetic tests pass; frozen certifier byte-identical (`science.py b564fba2` / `sealed.py 30ad6245`).

## GPU day - the one command

```bash
pip install torch torchvision            # CUDA build (replaces the +cpu wheel)
python scripts/prefetch_gpu_weights.py   # instant: weights already on disk, just re-verifies
ATTESTRA_DEVICE=cuda ATTESTRA_GPU_FT_ARCH=vit_l_16 python scripts/run_gpu_ceiling.py
```

The fine-tune arm is device-agnostic, autocast turns on automatically under CUDA, and the experiment certifies a
human-grade fine-tune against the frozen champion under the same discipline as every prior phase. The result lands
in `docs/GPU_CEILING_RESULT.json`.
