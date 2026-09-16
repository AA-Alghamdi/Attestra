"""Self-contained GPU smoke test run ON the pod via `runpodctl exec python`. No deps beyond torch (in the
stock runpod/pytorch image). Confirms cuda + a tiny on-GPU fit. Prints one JSON line marked VFRESULT."""
import json

out = {"cuda": False}
try:
    import torch
    import torch.nn as nn
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    out["device_name"] = torch.cuda.get_device_name(0) if out["cuda"] else "cpu"
    dev = "cuda" if out["cuda"] else "cpu"
    torch.manual_seed(0)
    X = torch.randn(256, 8, device=dev)
    y = (X.sum(1) > 0).long()
    net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lossf = nn.CrossEntropyLoss()
    for _ in range(60):
        opt.zero_grad(); lossf(net(X), y).backward(); opt.step()
    out["train_acc"] = round(float((net(X).argmax(1) == y).float().mean()), 3)
except Exception as e:  # noqa: BLE001
    out["error"] = f"{type(e).__name__}: {e}"
print("VFRESULT " + json.dumps(out))
