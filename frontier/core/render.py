"""frontier.core.render — NeuralSpec -> self-contained build_estimator() Program (design 03 §2).

``render_program(spec)`` turns a typed ``NeuralSpec`` into a Phase-0 ``Program`` whose ``code``
defines ``build_estimator()``. The generated code carries BOTH:
  - ``TorchNetEstimator``   : an import-gated real ``nn.Module`` + training loop (runs on the
                              pod where ``import torch`` succeeds);
  - ``SklearnMLPStandin``   : a local stand-in over sklearn's MLP (runs on the laptop, no torch).

``build_estimator()`` selects the branch AT RUNTIME inside the sandbox child, so the SAME
Program string rides the unmodified Task -> sandbox -> certify spine on both laptop and pod.

# === WIRING ===
The integrator never calls render.py directly; the NAS / LLM proposers (neural.py) call
``render_program`` and hand the resulting ``Program`` to the engine's proposer list. The engine
then sandbox-executes the code via ``frontier.sandbox.run_program`` exactly as it does an
sklearn recipe — no spine edit. Properties the spine relies on (CONTRACT.md + design 03 §2):

  1. SELF-CONTAINED. The generated module imports nothing from ``frontier``; the spec is inlined
     as a literal dict (``_SPEC_DICT``). This matches how ``proposers.make_code`` emits
     standalone sklearn modules and is required because the sandbox child execs the code in a
     bare namespace (sandbox.py ``_RUNNER``).
  2. EXACT INTERFACE. Both estimators implement only ``fit(X, y)`` / ``predict(X)`` — the two
     methods the sandbox runner calls. Neither returns a metric: the child writes predictions,
     the trusted parent scores (numeric firewall, CONTRACT invariant 2).
  3. RUNTIME BRANCH. ``build_estimator()`` tries ``import torch``; on success it builds the real
     net, on ImportError/any failure it falls back to the sklearn stand-in. The laptop therefore
     never executes torch code, and the pod runs the genuine network from the identical string.
  4. BACKEND HONESTY (CONTRACT invariant 5 + design 03 §9 "stand-in / torch divergence"). The
     chosen estimator stamps ``self.backend_`` ("torch" or "sklearn-standin") after fit, and the
     Program's ``provenance["backend_default"]`` records which branch the CURRENT host would take.
     A stand-in certificate is thus never conflated with a torch certificate: promotion of a
     torch architecture requires a torch-backend sealed certificate produced ON THE POD.

The stand-in is a FIDELITY-REDUCED surrogate (sklearn cannot express conv/attention): such
blocks are dropped with a recorded note in the realized estimator's ``dropped_blocks_``. The
stand-in's certificate licenses only the stand-in (design 03 §9). This is plumbing validation,
not a claim of architectural equivalence.
"""

from __future__ import annotations

import json

from ..program import Program
from .neural import NeuralSpec

# Determine, at RENDER time on the CURRENT host, which backend build_estimator() would select.
# This is provenance metadata only — the runtime branch in the generated code is what actually
# runs in the child. We never claim a torch run happened on a host without torch.
try:  # pragma: no cover - exercised on the pod
    import torch as _torch_probe  # noqa: F401
    _HOST_BACKEND = "torch"
except Exception:
    _HOST_BACKEND = "sklearn-standin"


def host_backend() -> str:
    """Return the backend the CURRENT host's render would default to ('torch'|'sklearn-standin')."""
    return _HOST_BACKEND


# The body of the generated module, with the __SPEC_JSON__ placeholder substituted by a plain
# str.replace (NOT str.format — the template contains many literal { } braces). Kept fully
# self-contained: no `import frontier`, no reference to the parent package. Both estimator
# classes are emitted so the child needs nothing but numpy + (torch on the pod | sklearn local).
_SPEC_PLACEHOLDER = "__FRONTIER_NEURAL_SPEC_JSON__"
_TEMPLATE = r'''# Auto-generated neural candidate (frontier.core.render) — fully self-contained.
# Rides the unmodified Phase-0 sandbox/certify spine. Returns predictions only (firewall).
import json
import numpy as np

# The NeuralSpec, inlined as a literal so the child needs no package import to load it.
_SPEC = json.loads(r"""__FRONTIER_NEURAL_SPEC_JSON__""")


def _spec_get(key, default=None):
    return _SPEC.get(key, default)


# --------------------------------------------------------------------------- sklearn stand-in
class SklearnMLPStandin:
    """Local, torch-free surrogate mapping the MLP-relevant parts of a NeuralSpec onto
    sklearn's MLPClassifier / MLPRegressor.

    conv1d / attention / embedding_pool blocks are DROPPED (sklearn cannot express them) and
    recorded in ``dropped_blocks_``. The stand-in is a fidelity-reduced development surrogate,
    NOT a claim of equivalence — its certificate licenses only the stand-in.
    """

    def __init__(self, spec):
        self.spec = spec
        self.backend_ = "sklearn-standin"
        self.dropped_blocks_ = []
        self.estimator_ = None
        self._label_dtype = None

    def _hidden_sizes(self):
        sizes = []
        for b in self.spec.get("blocks", []):
            if b.get("kind") in ("mlp", "residual_mlp"):
                sizes.append(int(b.get("width", 16)))
            else:
                self.dropped_blocks_.append(b.get("kind"))
        if not sizes:
            # at least one hidden layer so the stand-in is a real net, not a linear model
            sizes = [64]
        return tuple(sizes)

    def _activation(self):
        # sklearn supports {identity, logistic, tanh, relu}; map the spec's set to the closest.
        acts = [b.get("activation", "relu") for b in self.spec.get("blocks", [])
                if b.get("kind") in ("mlp", "residual_mlp")]
        a = acts[0] if acts else "relu"
        return {"relu": "relu", "gelu": "relu", "silu": "relu",
                "tanh": "tanh", "identity": "identity"}.get(a, "relu")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        kind = self.spec.get("task_kind", "classification")
        hidden = self._hidden_sizes()
        common = dict(
            hidden_layer_sizes=hidden,
            activation=self._activation(),
            alpha=float(self.spec.get("weight_decay", 1e-2)),
            learning_rate_init=float(self.spec.get("lr", 1e-3)),
            max_iter=int(self.spec.get("max_epochs", 200)),
            batch_size=min(int(self.spec.get("batch_size", 256)), max(1, len(X))),
            early_stopping=True,
            n_iter_no_change=int(self.spec.get("patience", 20)),
            validation_fraction=0.2,
            random_state=int(self.spec.get("seed", 0)),
        )
        if kind == "classification":
            from sklearn.neural_network import MLPClassifier
            self.estimator_ = MLPClassifier(**common)
            y = np.asarray(y).astype(str)   # matches the sandbox runner's label dtype
            self._label_dtype = "str"
        else:
            from sklearn.neural_network import MLPRegressor
            self.estimator_ = MLPRegressor(**common)
            y = np.asarray(y).astype(float)
            self._label_dtype = "float"
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = self.estimator_.predict(X)
        if self._label_dtype == "str":
            return np.asarray([str(p) for p in preds], dtype=object)
        return np.asarray([float(p) for p in preds], dtype=float)


# --------------------------------------------------------------------------- torch estimator
class TorchNetEstimator:
    """Import-gated real network. Built only when ``import torch`` succeeds (i.e. on the pod).

    Assembles the spec's blocks into an nn.Sequential spine with explicit residual wiring and,
    for sequence modality, a pooling head — driven by the spec, not a fixed template. Carries a
    self-contained training loop honoring the design-03 §3 protocol: inner-val early stopping
    (disjoint from the spine's val/sealed by construction — it only ever sees TRAIN rows),
    AdamW/Adam/SGD, cosine schedule with warmup, grad clipping, deterministic seeding, AMP only
    when CUDA is available, and a NaN/Inf guard that restores the last finite weights.

    Returns predictions only (string labels for clf, floats for reg). The parent scores.
    """

    def __init__(self, spec):
        self.spec = spec
        self.backend_ = "torch"
        self.dropped_blocks_ = []
        self._classes = None
        self.net_ = None

    # ---- module assembly ------------------------------------------------------------------
    def _build_module(self, in_features, torch, nn):
        spec = self.spec
        modality = spec.get("modality", "tabular")
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
                   "tanh": nn.Tanh, "identity": nn.Identity}

        class _ResidualWrap(nn.Module):
            """Wrap a block so its output is added to a (possibly projected) input."""
            def __init__(self, inner, proj):
                super().__init__()
                self.inner = inner
                self.proj = proj

            def forward(self, x):
                y = self.inner(x)
                s = self.proj(x) if self.proj is not None else x
                return y + s

        class _MeanPool(nn.Module):
            def forward(self, x):
                return x.mean(dim=1) if x.dim() == 3 else x

        layers = []
        cur = in_features
        is_seq = modality == "sequence"
        for b in spec.get("blocks", []):
            kind = b.get("kind")
            width = int(b.get("width", 64))
            act = act_map.get(b.get("activation", "relu"), nn.ReLU)
            norm = b.get("norm", "none")
            dropout = float(b.get("dropout", 0.0))
            if kind in ("mlp", "residual_mlp"):
                seq = [nn.Linear(cur, width)]
                if norm == "batch":
                    seq.append(nn.BatchNorm1d(width))
                elif norm == "layer":
                    seq.append(nn.LayerNorm(width))
                seq.append(act())
                if dropout > 0:
                    seq.append(nn.Dropout(dropout))
                inner = nn.Sequential(*seq)
                if b.get("residual") or kind == "residual_mlp":
                    proj = None if cur == width else nn.Linear(cur, width)
                    layers.append(_ResidualWrap(inner, proj))
                else:
                    layers.append(inner)
                cur = width
            elif kind == "conv1d" and is_seq:
                layers.append(nn.Conv1d(1 if cur == in_features else cur, width,
                                        kernel_size=int(b.get("kernel_size", 3)),
                                        stride=int(b.get("stride", 1))))
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(width))
                layers.append(act())
                cur = width
            elif kind == "attention" and is_seq:
                heads = int(b.get("n_heads", 4))
                layers.append(nn.MultiheadAttention(width, heads, batch_first=True))
                cur = width
            elif kind == "embedding_pool" and is_seq:
                layers.append(nn.Embedding(int(b.get("vocab_size", 2)), width))
                layers.append(_MeanPool())
                cur = width
            else:
                # unsupported in this assembly path on this modality -> record + skip
                self.dropped_blocks_.append(kind)
        layers.append(nn.Linear(cur, int(spec.get("out_dim", 1))))
        return nn.Sequential(*layers)

    def _inner_split(self, n, y, kind, rng):
        """Carve an inner-val slice out of TRAIN only (disjoint from spine val/sealed)."""
        import numpy as np
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(0.2 * n)))
        return idx[n_val:], idx[:n_val]

    def fit(self, X, y):
        import numpy as np
        import torch
        import torch.nn as nn
        spec = self.spec
        seed = int(spec.get("seed", 0))
        torch.manual_seed(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        rng = np.random.default_rng(seed)

        X = np.asarray(X, dtype=np.float32)
        kind = spec.get("task_kind", "classification")
        if kind == "classification":
            self._classes = sorted({str(v) for v in y})
            cls_index = {c: i for i, c in enumerate(self._classes)}
            y_idx = np.asarray([cls_index[str(v)] for v in y], dtype=np.int64)
            out_dim = len(self._classes)
        else:
            y_idx = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            out_dim = 1
        spec["out_dim"] = out_dim

        tr_idx, va_idx = self._inner_split(len(X), y_idx, kind, rng)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = self._build_module(X.shape[1], torch, nn).to(device)
        self.net_ = net

        Xtr = torch.as_tensor(X[tr_idx], device=device)
        Xva = torch.as_tensor(X[va_idx], device=device)
        if kind == "classification":
            ytr = torch.as_tensor(y_idx[tr_idx], device=device)
            yva = torch.as_tensor(y_idx[va_idx], device=device)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=float(spec.get("label_smoothing", 0.0)))
        else:
            ytr = torch.as_tensor(y_idx[tr_idx], device=device)
            yva = torch.as_tensor(y_idx[va_idx], device=device)
            loss_fn = nn.MSELoss()

        opt_name = spec.get("optimizer", "adamw")
        lr = float(spec.get("lr", 1e-3))
        wd = float(spec.get("weight_decay", 1e-2))
        if opt_name == "sgd":
            opt = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

        max_epochs = int(spec.get("max_epochs", 200))
        warmup = max(1, int(0.1 * max_epochs))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: (e + 1) / warmup if e < warmup
            else 0.5 * (1 + np.cos(np.pi * (e - warmup) / max(1, max_epochs - warmup))))
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        batch = max(1, min(int(spec.get("batch_size", 256)), len(Xtr)))
        patience = int(spec.get("patience", 20))
        best_va = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        last_finite = best_state
        no_improve = 0
        kill_floor_epoch = max(warmup + 1, int(0.25 * max_epochs))

        for epoch in range(max_epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=device)
            epoch_finite = True
            for s in range(0, len(Xtr), batch):
                sel = perm[s:s + batch]
                opt.zero_grad()
                with torch.autocast(device_type=("cuda" if use_amp else "cpu"), enabled=use_amp):
                    out = net(Xtr[sel])
                    loss = loss_fn(out, ytr[sel])
                if not torch.isfinite(loss):
                    epoch_finite = False
                    break
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sched.step()
            if not epoch_finite:
                net.load_state_dict(last_finite)  # restore last finite weights, then stop
                break
            last_finite = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.eval()
            with torch.no_grad():
                vout = net(Xva)
                vloss = float(loss_fn(vout, yva))
            if not np.isfinite(vloss):
                net.load_state_dict(best_state)
                break
            if vloss < best_va - 1e-6:
                best_va, no_improve = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                no_improve += 1
            # early-kill a hopeless run (design 03 innovation 2)
            if no_improve >= patience or (epoch >= kill_floor_epoch and best_va == float("inf")):
                break
        net.load_state_dict(best_state)
        self._device = device
        return self

    def predict(self, X):
        import numpy as np
        import torch
        X = np.asarray(X, dtype=np.float32)
        net = self.net_
        net.eval()
        with torch.no_grad():
            out = net(torch.as_tensor(X, device=self._device))
            out = out.detach().cpu().numpy()
        if self.spec.get("task_kind", "classification") == "classification":
            idx = out.argmax(axis=1)
            return np.asarray([self._classes[i] for i in idx], dtype=object)
        return np.asarray(out.reshape(-1), dtype=float)


def build_estimator():
    """Return a fitted-on-fit estimator. Branch chosen at runtime so the same code is portable.

    torch present (pod) -> the genuine network; torch absent (laptop) -> the sklearn stand-in.
    """
    try:
        import torch  # noqa: F401
        return TorchNetEstimator(_SPEC)
    except Exception:
        return SklearnMLPStandin(_SPEC)
'''


def render_program(spec: NeuralSpec) -> Program:
    """Render a ``NeuralSpec`` to a self-contained ``Program`` (design 03 §2).

    The returned Program's ``code`` defines ``build_estimator()`` and imports nothing from
    frontier. ``provenance`` carries the full spec dict (so NASProposer can recover the champion
    spec from the winning Program — design 03 §6) and the host-default backend tag (so a stand-in
    certificate is never conflated with a torch certificate — design 03 §9 / CONTRACT inv. 5).
    """
    if not isinstance(spec, NeuralSpec):
        raise TypeError("render_program expects a NeuralSpec")
    spec_json = spec.to_json()
    # guard: the inlined JSON must not contain the triple-quote sequence that would close the
    # heredoc. JSON never emits a bare \"\"\" but be defensive.
    if '"""' in spec_json:
        spec_json = spec_json.replace('"""', '\\"\\"\\"')
    code = _TEMPLATE.replace(_SPEC_PLACEHOLDER, spec_json)

    author = spec.provenance.get("author", "nas") if spec.provenance else "nas"
    source = "llm" if author == "llm" else "nas"
    label = f"net_{spec.modality[:3]}_{spec.fingerprint}"
    spec_dict = spec.to_dict()
    return Program(
        code=code,
        source=source,
        label=label,
        provenance={
            "neural_spec": spec_dict,
            # NASProposer reads best_recipe.neural_spec as a fallback recovery path; mirror it.
            "recipe": {"neural_spec": spec_dict},
            "backend_default": host_backend(),
            "author": author,
        },
    )
