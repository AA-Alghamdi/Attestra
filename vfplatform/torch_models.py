"""Device-agnostic torch MLP wrappers (sklearn-style fit/predict) for the GPU harness.

The SAME code runs on CPU locally and on `cuda` in the RunPod worker (device is auto-detected), so the
GPU harness is testable on the laptop and drops straight onto the GPU. These plug into the existing loop
and featurizer unchanged because they expose .fit(X, y) / .predict(X).
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def torch_device():
    if not _HAS_TORCH:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_dense(X):
    if hasattr(X, "toarray"):           # scipy sparse (text tfidf) -> dense for the MLP
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


class _MLP(nn.Module if _HAS_TORCH else object):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _TorchBase:
    def __init__(self, hidden=(64, 64), dropout=0.1, lr=1e-3, epochs=60, weight_decay=0.0, seed=0):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.epochs, self.weight_decay, self.seed = epochs, weight_decay, seed
        self.device = torch_device()
        self.model = None

    def _train(self, X, y, out_dim, loss_fn, y_dtype):
        torch.manual_seed(self.seed)
        X = _to_dense(X)
        xt = torch.tensor(X, device=self.device)
        yt = torch.tensor(np.asarray(y), dtype=y_dtype, device=self.device)
        self.model = _MLP(X.shape[1], self.hidden, out_dim, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()
        n = X.shape[0]
        bs = min(256, n)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                out = self.model(xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
        return self


class TorchMLPClassifier(_TorchBase):
    def __init__(self, n_classes=2, **kw):
        super().__init__(**kw)
        self.n_classes = n_classes

    def fit(self, X, y):
        self.n_classes = int(max(self.n_classes, int(np.max(y)) + 1))
        return self._train(X, y, self.n_classes, nn.CrossEntropyLoss(), torch.long)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(_to_dense(X), device=self.device))
            return out.argmax(1).cpu().numpy()


class TorchMLPRegressor(_TorchBase):
    def fit(self, X, y):
        return self._train(X, np.asarray(y, dtype=np.float32).reshape(-1, 1), 1,
                           nn.MSELoss(), torch.float32)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(_to_dense(X), device=self.device))
            return out.squeeze(-1).cpu().numpy()
