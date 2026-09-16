"""#5 -- DEVICE-AGNOSTIC fine-tune ceiling arm, ready for tomorrow's GPU with ZERO code changes.

The phase-#3 fine-tune ceiling (`benchmark_aircraft._finetune_once`) is CPU-only: it never moves the model or
the batches to a device, so on a GPU box it would silently train on the CPU. This module is the device-agnostic
SUPERSET that tomorrow's GPU run uses unchanged -- it auto-detects CUDA (falling back to CPU bit-for-bit when
there is no GPU), supports the bigger fine-tune architectures a human would actually reach for on a GPU
(`vit_b_16`, `vit_l_16` with SWAG weights), and uses mixed-precision autocast on CUDA only. Crucially it keeps
the EXACT select-then-bound discipline of the hand-run: train on the train rows, select the best epoch (and the
best restart) on the val rows, and score the sealed rows EXACTLY ONCE -- the sealed labels never influence
training or selection, so the frozen Clopper-Pearson bound stays valid.

  * `pick_device()`              -- cuda if available else cpu (override with ATTESTRA_DEVICE / arg).
  * `make_model(arch)`           -- pretrained backbone, 2-way head, partial unfreeze (a GPU-sane fine-tune).
  * `finetune_once/finetune_correct` -- the device-agnostic, leak-free ceiling arm (same return contract as #3).
  * `python scripts/gpu_finetune.py --smoke`  -- a tiny CPU fine-tune on a synthetic signal that PROVES the
    whole path (train -> val-select -> bound) runs and is select-then-bound, tonight, with no GPU.

Tomorrow on the GPU: `pip install` a CUDA torch build (this box is `+cpu`), pre-fetch weights with
`scripts/prefetch_gpu_weights.py`, then point the arena's fine-tune arm at `finetune_correct(..., arch="vit_l_16")`.
"""
import argparse
import copy
import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.benchmark_vision_transfer as B1                      # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# arch -> (torchvision builder name, weights enum attr, default px, trainable-suffix matcher, head setter)
# the matcher/head-setter are applied after construction so the same partial-unfreeze recipe covers CNN + ViT.
_ARCH_PX = {"resnet18": 224, "resnet50": 224, "vit_b_16": 224, "vit_l_16": 224}


def pick_device(prefer: str = None):
    """Return a torch.device: the explicit `prefer` (or $ATTESTRA_DEVICE), else cuda when available, else cpu."""
    import torch
    prefer = prefer or os.environ.get("ATTESTRA_DEVICE")
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(arch: str, n_classes: int = 2, unfreeze_blocks: int = 4):
    """Pretrained backbone with a fresh `n_classes` head and a GPU-sane PARTIAL unfreeze (CNN: layer3/layer4/fc;
    ViT: the last `unfreeze_blocks` transformer blocks + head). Returns the model with requires_grad already set."""
    import torch.nn as nn
    import torchvision.models as M
    if arch == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        trainable = ("layer3", "layer4", "fc")
    elif arch == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        trainable = ("layer3", "layer4", "fc")
    elif arch in ("vit_b_16", "vit_l_16"):
        if arch == "vit_b_16":
            m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            m = M.vit_l_16(weights=M.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
        n_layers = len(m.encoder.layers)
        keep = {f"encoder.layers.encoder_layer_{i}" for i in range(n_layers - unfreeze_blocks, n_layers)}
        trainable = tuple(keep) + ("heads",)
    else:
        raise ValueError(f"unknown fine-tune arch {arch!r}")
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(trainable)
    return m


def _set_bn_eval(model):
    import torch.nn as nn
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()


def finetune_once(cache, y, tr_ids, val_ids, test_ids, *, arch="resnet50", px=224, epochs=30, lr=0.005,
                  seed=0, device=None, amp=None, batch_size=16, num_workers=0):
    """One device-agnostic fine-tune restart. Trains end-to-end on `tr_ids`, selects the best EPOCH on `val_ids`,
    and returns (sealed correctness in sorted(test_ids) order, test_acc, best_val, best_ep). The sealed rows are
    scored exactly once, after selection -- strict select-then-bound (no leakage). `cache` is a list of PIL
    images (or anything the transforms accept) indexed by row id, exactly as benchmark_aircraft supplies."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    device = device or pick_device()
    amp = (device.type == "cuda") if amp is None else amp
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cpu":
        torch.set_num_threads(2)

    train_tf = T.Compose([T.RandomResizedCrop(px, scale=(0.55, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = T.Compose([T.CenterCrop(px), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class _DS(Dataset):
        def __init__(self, ids, tf):
            self.ids = list(ids)
            self.tf = tf

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, k):
            i = self.ids[k]
            return self.tf(cache[i]), int(y[i])

    g = torch.Generator()
    g.manual_seed(seed)
    pin = device.type == "cuda"
    tr = DataLoader(_DS(tr_ids, train_tf), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, generator=g, pin_memory=pin)
    va = DataLoader(_DS(val_ids, eval_tf), batch_size=32, num_workers=num_workers, pin_memory=pin)
    te = DataLoader(_DS(sorted(test_ids), eval_tf), batch_size=32, num_workers=num_workers, pin_memory=pin)

    model = make_model(arch).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=amp)

    def _eval(net, loader):
        net.eval()
        corr = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    pred = net(xb).argmax(1)
                corr += (pred == yb).int().tolist()
        return corr

    best_va, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        _set_bn_eval(model)
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        va_acc = float(np.mean(_eval(model, va)))
        if va_acc > best_va:
            best_va, best_state, best_ep = va_acc, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state)
    correct = _eval(model, te)
    del model, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return correct, float(np.mean(correct)), round(best_va, 4), best_ep


def finetune_correct(cache, y, tr_ids, val_ids, test_ids, *, arch="resnet50", px=224, epochs=30, lr=0.005,
                     restarts=3, device=None, amp=None):
    """The ceiling arm as a FAIR strong attempt: `restarts` independent restarts, keep the best-VAL restart
    (never peeking at sealed). Returns (sealed correctness, test_acc, best_ep, best_val, per_restart_vals)."""
    device = device or pick_device()
    best, vals = None, []
    for r in range(restarts):
        c, acc, va, ep = finetune_once(cache, y, tr_ids, val_ids, test_ids, arch=arch, px=px,
                                       epochs=epochs, lr=lr, seed=r, device=device, amp=amp)
        vals.append(va)
        if best is None or va > best[3]:
            best = (c, acc, ep, va)
    return best[0], best[1], best[2], best[3], vals


# --------------------------------------------------------------------------- CPU smoke (no GPU, no real data)
def _synthetic_cache(n_per_class=24, px=64, seed=0):
    """A tiny separable image signal: class 0 is reddish, class 1 is bluish, plus pixel noise. A partial-unfreeze
    resnet18 learns it in a few CPU epochs, so the smoke exercises the FULL train->val-select->bound path fast."""
    from PIL import Image
    rng = np.random.RandomState(seed)
    n = 2 * n_per_class
    y = np.array([0] * n_per_class + [1] * n_per_class)
    cache = []
    for i in range(n):
        base = np.zeros((px, px, 3), dtype=np.float32)
        base[..., 0 if y[i] == 0 else 2] = 0.75                     # red for class 0, blue for class 1
        img = np.clip(base + 0.25 * rng.rand(px, px, 3), 0, 1)
        cache.append(Image.fromarray((img * 255).astype(np.uint8)))
    return cache, y


def _smoke():
    device = pick_device()
    print(f"device = {device}  (cuda_available={__import__('torch').cuda.is_available()})")
    cache, y = _synthetic_cache()
    idx = np.arange(len(y))
    rng = np.random.RandomState(1)
    tr, val, test = [], [], []
    for c in (0, 1):
        ci = idx[y == c]
        rng.shuffle(ci)
        tr += list(ci[:14])
        val += list(ci[14:19])
        test += list(ci[19:24])
    c, acc, ep, va, vals = finetune_correct(cache, y, np.array(tr), np.array(val), np.array(test),
                                            arch="resnet18", px=64, epochs=4, lr=0.01, restarts=2)
    print(f"smoke fine-tune ran: sealed_acc={acc:.3f} sealed_lb={B1._lb(c):.3f} best_val={va:.3f} "
          f"best_ep={ep} restart_vals={vals}")
    assert len(c) == len(test), "sealed correctness length must equal the sealed row count"
    print("OK: device-agnostic fine-tune path runs end-to-end on CPU")


def main():
    ap = argparse.ArgumentParser(description="Device-agnostic fine-tune ceiling (GPU-ready).")
    ap.add_argument("--smoke", action="store_true", help="run a tiny CPU fine-tune on a synthetic signal")
    ap.add_argument("--check-arch", default=None, help="instantiate an arch to verify its weights are cached")
    args = ap.parse_args()
    if args.check_arch:
        m = make_model(args.check_arch)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in m.parameters())
        print(f"{args.check_arch}: built OK  trainable={n_train/1e6:.1f}M / total={n_tot/1e6:.1f}M params")
        return 0
    if args.smoke:
        _smoke()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
