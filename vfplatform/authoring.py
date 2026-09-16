"""Tier-2: sandboxed LLM-CODE-AUTHORING -- the LLM emits a novel sklearn-compatible estimator/featurizer
as executable Python, which (after a frozen three-stage admission gate) becomes just ANOTHER candidate Move
flowing through the EXISTING frozen certifier. The LLM is NON-BINDING: it never promotes.

!!! SECURITY STATUS (2026-06-17 adversarial pass) -- RESEARCH PROTOTYPE, NOT WIRED INTO THE LOOP !!!
An adversarial security pass BROKE the original AST-allowlist sandbox: because numpy is on the import
allowlist and numpy's C-extension file I/O (np.savetxt/save/load/fromfile/genfromtxt/memmap/ctypeslib) needs
NONE of the banned names, an admitted estimator read and EXFILTRATED the .anthropic_key during admission, and
the in-process certify/fit path runs authored code with the parent's full filesystem + no rlimit/timeout.
HARDENING APPLIED HERE: those numpy I/O attrs + frame/code introspection attrs + bare __builtins__ are now
AST-denied (closes the CONFIRMED vectors). BUT an AST allowlist is necessary-NOT-sufficient for numpy/C
extensions, and macOS does not enforce RLIMIT_AS (mem-bomb residual). THEREFORE: this module is NOT imported
or wired by loop.py and MUST NOT be wired into the live search until ALL authored-code execution (admission
AND every fit/predict) runs under OS-level isolation (container / seccomp / no-fs namespace). Treat any call
to admit() on untrusted code as running it in-process; only run it on a host you can throw away. Admission is a
SAFETY + BUILDABILITY + CONTRACT check, NOT a quality judgement; quality is decided ONLY by the loop's
validation selection and the frozen sealed certify path (vectorforge/science.py b564fba2,
vfplatform/sealed.py 30ad6245 -- both re-verified unchanged this session, never edited here).
select-then-bound keeps the certificate at checks=1 no matter how many authored candidates are searched.

This is a NEW, ISOLATED module. It does NOT import or alter loop.py, harness's move executor, sealed.py,
or science.py. It reuses the proven A8 connector authoring AST gate (vectorforge/llm_shell/authoring/
static_gate.py) verbatim for the banned-name / banned-attr machinery, and adds an estimator-specific import
allowlist (numpy / sklearn / math) + a build_estimator entrypoint contract.

THREE-STAGE ADMISSION GATE (all three must pass; each failure rejects with a SPECIFIC, auditable reason):

  Stage 1 -- STATIC AST GATE  (estimator_static_check)
     import allowlist {numpy, math, sklearn (+ a sklearn-submodule allowlist)} ONLY; sklearn.datasets and
     sklearn.model_selection are explicitly DENIED (a method has no business loading data or peeking at a
     split); the frozen A8 BANNED_NAMES (eval/exec/compile/open/__import__/getattr/...) and BANNED_ATTRS (all
     dunder introspection escapes) are inherited; a defense-in-depth identifier denylist (os/sys/subprocess/
     socket/pickle/...) is added; `build_estimator(seed)` arity is forced to 1; constant-true `while` loops
     without a `break` are rejected (infinite-loop guard); global/nonlocal rejected; AST node count capped.

  Stage 2 -- ISOLATED EXECUTION  (run_isolated)
     a FRESH interpreter via multiprocessing 'spawn' (NOT fork) -> the child carries none of the parent's
     loaded modules / open fds / the in-memory sealed test (structural sealed-blindness for free). The child
     installs resource.setrlimit(RLIMIT_CPU) (ENFORCED, portable) and RLIMIT_AS/RLIMIT_DATA (ENFORCED on
     Linux/RunPod, BEST-EFFORT only on Darwin where the kernel does not honor RLIMIT_AS -- documented, NOT
     overclaimed), execs the source under restricted builtins + a restricted __import__, and runs the task.
     The PARENT enforces a WALL-CLOCK timeout via Pipe.poll(wall_s) + terminate()->kill(): this is the
     universal backstop that catches anything the rlimits miss (sleep, deadlock, mem bomb on Darwin).

  Stage 3 -- SCIENTIFIC SELF-TEST  (scientific_selftest, runs ENTIRELY inside the isolated child)
     on a tiny SYNTHETIC dataset the gate fabricates deterministically (never the real task data, never the
     sealed test): (a) VALID PREDICTIONS -- predict length == n_rows, classifier labels in range(n_classes),
     regressor finite floats; (b) DETERMINISM -- two builds with the SAME seed give byte-identical
     predictions; (c) NO SEALED-TEST ACCESS -- structural (spawn child has no channel to the sealed rows;
     static gate forbids datasets/IO) PLUS a behavioral permutation-equivariance probe (predict(X[perm]) ==
     predict(X)[perm]) that catches index-based leakage; (d) TRAIN-ONLY FIT on a disjoint holdout.

HONEST LIMIT (not overclaimed): static gate + spawned process + restricted exec is a STRONG control against
accidental and most adversarial code, but it is NOT a seccomp/gVisor/container jail. A novel CPython exploit
reachable purely through the numpy/sklearn C extensions (no banned import, no dunder access) is out of scope
of the AST gate. The authored method is build/select-time, NEVER on the live request path; for fully
untrusted authoring at scale, run the child inside the seccomp'd RunPod worker container. Same caveat the A8
connector path documents. On Darwin, RLIMIT_AS is unenforced; the parent wall-clock timeout is the enforced
backstop there (memory bomb -> wall timeout).
"""
import ast
import math
import numbers
import os
import re
import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vectorforge.llm_shell import ops
from vectorforge.llm_shell.authoring import static_gate as _sg   # reuse the frozen A8 AST gate machinery
from .harness import CatalogEntry, move_from_proposal, _grid_configs

SURFACE = "vfplatform.authoring.estimator"
PROMPT_VERSION = "authored-estimator/v3"
_TOOL_NAME = "emit_estimator"
ENTRYPOINT = "build_estimator"

# Estimators legitimately need numpy + sklearn + math; nothing else. This is a DIFFERENT, estimator-specific
# allowlist (we do NOT reuse the connector json/re/csv allowlist).
ALLOWED_IMPORT_ROOTS = {"numpy", "np", "sklearn", "math"}
# sklearn submodules a method may import. A method has NO business loading data or peeking at a split.
ALLOWED_SKLEARN_SUBMODULES = {
    "sklearn", "sklearn.linear_model", "sklearn.ensemble", "sklearn.svm", "sklearn.tree",
    "sklearn.neighbors", "sklearn.preprocessing", "sklearn.base", "sklearn.utils",
    "sklearn.metrics", "sklearn.kernel_approximation", "sklearn.decomposition",
    "sklearn.pipeline", "sklearn.multiclass", "sklearn.naive_bayes", "sklearn.cluster",
    "sklearn.gaussian_process", "sklearn.neural_network", "sklearn.discriminant_analysis",
}
DENIED_SKLEARN_SUBMODULES = {"sklearn.datasets", "sklearn.model_selection", "sklearn.externals"}

# Inherited verbatim from the frozen A8 connector gate.
BANNED_NAMES = set(_sg.BANNED_NAMES)
BANNED_ATTRS = set(_sg.BANNED_ATTRS)
# Defense-in-depth identifier denylist: even a string-built reference is conspicuous.
DENYLIST_IDENTIFIERS = {
    "os", "sys", "subprocess", "socket", "builtins", "importlib", "ctypes", "pickle",
    "marshal", "shutil", "pathlib", "requests", "urllib", "http", "io", "tempfile",
    "threading", "asyncio", "ftplib", "smtplib", "signal", "resource", "multiprocessing",
    "__builtins__",   # adversarial finding: a bare __builtins__ Name slipped the dunder-ATTRIBUTE check
}
# Adversarial-pass finding: numpy is on the import allowlist, but numpy's OWN C-extension file I/O needs NONE
# of the banned names (no `open`, no `os`) -- np.savetxt/save/load/fromfile/genfromtxt/memmap/ctypeslib read &
# write arbitrary host files and were used to exfiltrate the .anthropic_key during admission. Deny these
# attributes (and frame/code introspection attrs) at the AST layer. NOTE: this closes the CONFIRMED vectors but
# the AST gate is necessary-not-sufficient for numpy/C-extensions in general (see the SECURITY banner up top).
DENIED_RUNTIME_ATTRS = {
    "savetxt", "save", "savez", "savez_compressed", "load", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "ctypeslib", "fromregex", "DataSource", "frombuffer", "getbuffer", "fromiter",
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "gi_frame", "gi_code", "cr_frame",
}

_MAX_CODE_CHARS = 8000
_MAX_AST_NODES = 4000
_DEF_CPU_S = 8
_DEF_MEM_MB = 1024
_DEF_WALL_S = 25.0


# ===================================================================== Stage 1: static AST gate
@dataclass
class EstimatorStaticReport:
    ok: bool
    violations: list = field(default_factory=list)
    has_entrypoint: bool = False

    def as_dict(self):
        return {"ok": self.ok, "violations": list(self.violations),
                "has_entrypoint": self.has_entrypoint}


def _is_constant_true(node):
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.Name) and node.id == "True":
        return True
    return False


def _has_break(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Break):
            return True
    return False


def estimator_static_check(code: str, *, entrypoint: str = ENTRYPOINT) -> EstimatorStaticReport:
    """STRICT AST allowlist/denylist tuned for an estimator/featurizer factory. Mirrors the frozen A8
    connector gate but with the numpy/sklearn/math allowlist, a sklearn-submodule allow/deny, a
    `build_estimator(seed)`-arity check, an infinite-loop guard, and an AST-size cap. Rejects
    os/sys/network/file/import-tricks/dunder introspection so only pure numerical model code runs.
    Never executes anything."""
    if not isinstance(code, str) or not code.strip():
        return EstimatorStaticReport(False, ["empty code"])
    if len(code) > _MAX_CODE_CHARS:
        return EstimatorStaticReport(False, [f"code exceeds {_MAX_CODE_CHARS} chars"])
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        return EstimatorStaticReport(False, [f"syntax error: {ex}"])

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        return EstimatorStaticReport(False, [f"method too large to gate ({len(nodes)} AST nodes > {_MAX_AST_NODES})"])

    v, has_entry = [], False
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                full = a.name
                root = full.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    v.append(f"import of {full!r} not allowed (allowlist {sorted(ALLOWED_IMPORT_ROOTS)})")
                elif root == "sklearn" and full in DENIED_SKLEARN_SUBMODULES:
                    v.append(f"import of {full!r} explicitly denied (data/split access)")
                elif root == "sklearn" and full not in ALLOWED_SKLEARN_SUBMODULES:
                    v.append(f"sklearn submodule {full!r} not in allowlist")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if node.level != 0:
                v.append("relative import not allowed")
            elif root not in ALLOWED_IMPORT_ROOTS:
                v.append(f"import from {mod!r} not allowed (allowlist {sorted(ALLOWED_IMPORT_ROOTS)})")
            elif root == "sklearn" and mod in DENIED_SKLEARN_SUBMODULES:
                v.append(f"import from {mod!r} explicitly denied (data/split access)")
            elif root == "sklearn" and mod not in ALLOWED_SKLEARN_SUBMODULES:
                v.append(f"sklearn submodule {mod!r} not in allowlist")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                v.append(f"use of banned name {node.id!r}")
            elif node.id in DENYLIST_IDENTIFIERS:
                v.append(f"use of denied identifier {node.id!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS or (node.attr.startswith("__") and node.attr.endswith("__")):
                v.append(f"access to forbidden attribute {node.attr!r}")
            elif node.attr in DENIED_RUNTIME_ATTRS:
                v.append(f"access to denied runtime attribute {node.attr!r} "
                         f"(numpy/C-extension file I/O or frame introspection)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            v.append("global/nonlocal not allowed")
        elif isinstance(node, ast.While):
            if _is_constant_true(node.test) and not _has_break(node):
                v.append("constant-true while loop without break (infinite-loop guard)")
        elif isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            has_entry = True
            n_args = len(node.args.args)
            if n_args != 1:
                v.append(f"entrypoint {entrypoint!r} must take exactly 1 arg (seed), has {n_args}")

    if not has_entry:
        v.append(f"no entrypoint function named {entrypoint!r}")
    return EstimatorStaticReport(ok=(len(v) == 0), violations=v, has_entrypoint=has_entry)


# ===================================================================== the authored-estimator contract
@dataclass
class EstimatorSpec:
    """What the LLM authors against. The self-test data is derived from this shape -- NOT from any target
    paper's numbers. role='classifier'|'regressor'|'featurizer'."""
    role: str                          # classifier | regressor | featurizer
    n_features: int = 8
    n_classes: int = 2                 # >=2 for classifier; 1 for regressor/featurizer
    family_prefix: str = "authored"
    entrypoint: str = ENTRYPOINT
    # resource limits for admission (Stage 2)
    cpu_s: int = _DEF_CPU_S
    mem_mb: int = _DEF_MEM_MB
    wall_s: float = _DEF_WALL_S

    def as_payload(self):
        return {"role": self.role, "n_features": int(self.n_features),
                "n_classes": int(self.n_classes), "entrypoint": self.entrypoint}


# ===================================================================== restricted in-process exec
# Built lazily so the module imports on a machine without sklearn installed at import time.
_SAFE_BUILTINS = None


def _build_safe_builtins():
    import builtins as _b
    keep = ("len", "range", "enumerate", "str", "int", "float", "bool", "list", "dict", "tuple",
            "set", "frozenset", "sorted", "reversed", "sum", "min", "max", "abs", "round", "zip",
            "map", "filter", "any", "all", "isinstance", "issubclass", "repr", "ord", "chr", "slice",
            "next", "iter", "divmod", "pow", "hash", "id", "print", "format",
            "ValueError", "KeyError", "IndexError", "TypeError", "ZeroDivisionError",
            "ArithmeticError", "Exception", "StopIteration", "RuntimeError", "AttributeError",
            "NotImplementedError", "OverflowError", "FloatingPointError")
    safe = {k: getattr(_b, k) for k in keep}
    # `class` statements compile to a __build_class__ call; it is a pure language primitive (no FS/OS/net)
    # and is required for any authored estimator that defines a class. Safe to expose.
    safe["__build_class__"] = _b.__build_class__

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        full = str(name)
        root = full.split(".")[0]
        if level != 0:
            raise ImportError("relative import not permitted in the authoring sandbox")
        if root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"import of {full!r} not permitted in the authoring sandbox")
        if root == "sklearn" and full in DENIED_SKLEARN_SUBMODULES:
            raise ImportError(f"import of {full!r} explicitly denied (data/split access)")
        return __import__(name, globals, locals, fromlist, level)

    safe["__import__"] = _safe_import
    return safe


def _compile_build(code, entrypoint=ENTRYPOINT):
    """Exec statically-gated estimator code in a restricted namespace; return (ok, build_fn, err). Callers
    MUST run the static gate first. Unlike the connector sandbox (fixed pre-injected modules) we permit
    allowlisted real __import__ at module root so `from sklearn.ensemble import ...` resolves. Never raises."""
    global _SAFE_BUILTINS
    if _SAFE_BUILTINS is None:
        _SAFE_BUILTINS = _build_safe_builtins()
    ns = {"__builtins__": _SAFE_BUILTINS, "__name__": "authored_estimator"}
    try:
        exec(compile(code, "<authored_estimator>", "exec"), ns)  # noqa: S102 statically-gated, restricted ns
    except Exception as ex:  # noqa: BLE001
        return False, None, f"exec error: {type(ex).__name__}: {str(ex)[:160]}"
    fn = ns.get(entrypoint)
    if not callable(fn):
        return False, None, f"entrypoint {entrypoint!r} not callable"
    return True, fn, None


class _AuthoredWrapper:
    """Normalize an authored object to the sklearn surface the loop expects. A classifier/regressor must
    already expose fit/predict (passthrough). A featurizer (fit/transform) is paired with a FROZEN linear
    head (LogisticRegression for classification, Ridge for regression) so it ALSO presents fit/predict and
    flows through the identical execute path. No predict_proba required."""
    def __init__(self, est, role):
        self.role = role
        if role in ("classifier", "regressor"):
            if not (hasattr(est, "fit") and hasattr(est, "predict")):
                raise TypeError("authored estimator must expose fit/predict")
            self._est = est
            self._feat = self._head = None
        elif role == "featurizer":
            from sklearn.linear_model import LogisticRegression
            if not (hasattr(est, "fit") and hasattr(est, "transform")):
                raise TypeError("authored featurizer must expose fit/transform")
            self._feat, self._head, self._est = est, LogisticRegression(max_iter=2000), None
        elif role == "featurizer_reg":
            from sklearn.linear_model import Ridge
            if not (hasattr(est, "fit") and hasattr(est, "transform")):
                raise TypeError("authored featurizer must expose fit/transform")
            self._feat, self._head, self._est = est, Ridge(), None
        else:
            raise TypeError(f"unknown role {role!r}")

    def fit(self, X, y):
        if self._est is not None:
            self._est.fit(X, y)
            return self
        Z = self._feat.fit(X, y).transform(X) if _accepts_y(self._feat) else self._feat.fit(X).transform(X)
        self._head.fit(np.asarray(Z), y)
        return self

    def predict(self, X):
        if self._est is not None:
            return self._est.predict(X)
        return self._head.predict(np.asarray(self._feat.transform(X)))


def _accepts_y(feat):
    import inspect
    try:
        return len(inspect.signature(feat.fit).parameters) >= 2
    except (TypeError, ValueError):
        return True


# ===================================================================== Stage 2: isolated execution
def _make_synth(spec_payload, seed=0):
    """Deterministic tiny synthetic dataset matching the contract shape. NEVER the real task data, NEVER the
    sealed test. Returns (Xtr, ytr, Xte, yte)."""
    role = spec_payload["role"]
    nf = int(spec_payload["n_features"])
    nc = int(spec_payload["n_classes"])
    rng = np.random.RandomState(seed)
    n = 160
    X = rng.randn(n, nf).astype(float)
    if role == "regressor" or role == "featurizer_reg":
        w = rng.randn(nf)
        y = (X @ w + 0.1 * rng.randn(n)).astype(float)
    else:
        ncc = max(2, nc)
        # a learnable signal so a real classifier can fit; labels in [0, ncc)
        scores = X[:, : min(nf, ncc)]
        if scores.shape[1] < ncc:
            scores = np.hstack([scores, rng.randn(n, ncc - scores.shape[1])])
        y = scores.argmax(axis=1).astype(int)
    split = 112
    return X[:split], y[:split], X[split:], y[split:]


def _child_main(conn, source, task, spec_payload, cpu_s, mem_mb):
    """Spawn target (top-level, picklable). FRESH interpreter: none of the parent's modules/fds/sealed test.
    Installs rlimits, hardens the environment, execs the source under restricted builtins, runs `task`, and
    sends ('ok', result) or ('error', reason) back through the Pipe. Never lets an exception escape."""
    try:
        import signal as _signal
        import resource as _resource

        # ---- CPU cap (ENFORCED, portable): SIGXCPU at soft, SIGKILL at hard.
        def _on_xcpu(signum, frame):
            try:
                conn.send(("error", f"cpu timeout ({cpu_s}s)"))
                conn.close()
            finally:
                os._exit(1)
        try:
            _signal.signal(_signal.SIGXCPU, _on_xcpu)
            _resource.setrlimit(_resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 2))
        except (ValueError, OSError, AttributeError):
            pass

        # ---- Memory cap (ENFORCED on Linux; BEST-EFFORT on Darwin where RLIMIT_AS is unenforced -- the
        #      parent wall-clock timeout is the backstop there). Set generously above sklearn's own need.
        nbytes = int(mem_mb) * 1024 * 1024
        for lim in ("RLIMIT_AS", "RLIMIT_DATA"):
            try:
                r = getattr(_resource, lim)
                soft, hard = _resource.getrlimit(r)
                newhard = hard if hard != _resource.RLIM_INFINITY and hard < nbytes else nbytes
                _resource.setrlimit(r, (nbytes, newhard))
            except (ValueError, OSError, AttributeError):
                pass

        # ---- harden the environment: no inherited secrets, empty cwd, no fs writes expected.
        try:
            for k in list(os.environ.keys()):
                if "KEY" in k.upper() or "TOKEN" in k.upper() or "SECRET" in k.upper():
                    del os.environ[k]
        except Exception:  # noqa: BLE001
            pass

        ok, build_fn, err = _compile_build(source, ENTRYPOINT)
        if not ok:
            conn.send(("error", err))
            conn.close()
            return

        if task == "selftest":
            res = _run_selftest(build_fn, spec_payload)
            conn.send(("ok", res))
        elif task == "fit_predict":
            role = spec_payload["role"]
            Xtr, ytr, Xte, _ = _make_synth(spec_payload, seed=0)
            est = _AuthoredWrapper(build_fn(0), role)
            est.fit(Xtr, ytr)
            pred = np.asarray(est.predict(Xte))
            conn.send(("ok", {"n": int(pred.shape[0]), "finite": bool(np.all(np.isfinite(pred.astype(float))))}))
        else:
            conn.send(("error", f"unknown task {task!r}"))
        conn.close()
    except MemoryError:
        try:
            conn.send(("error", "memory cap exceeded (MemoryError)"))
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(1)
    except Exception as ex:  # noqa: BLE001
        try:
            conn.send(("error", f"{type(ex).__name__}: {str(ex)[:200]}"))
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def run_isolated(source, task, spec_payload, *, cpu_s=_DEF_CPU_S, mem_mb=_DEF_MEM_MB, wall_s=_DEF_WALL_S):
    """Run `task` on statically-gated `source` in a FRESH spawned interpreter with rlimits + a parent
    wall-clock backstop. Returns (ok, result, reason). Never raises. Callers MUST run the static gate first."""
    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = multiprocessing.get_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_main, args=(child_conn, source, task, spec_payload, cpu_s, mem_mb))
    p.start()
    child_conn.close()
    got = parent_conn.poll(wall_s)          # WALL clock is the universal backstop (catches mem bomb on mac)
    if not got:
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
            p.join(2)
        parent_conn.close()
        return (False, None, f"wall timeout {wall_s}s")
    try:
        status, payload = parent_conn.recv()
    except EOFError:
        parent_conn.close()
        p.join(2)
        return (False, None, f"child died without a reply (rlimit kill / crash; exit {p.exitcode})")
    parent_conn.close()
    p.join(2)
    if p.is_alive():
        p.kill()
        p.join(2)
    if status == "ok":
        return (True, payload, None)
    return (False, None, str(payload))


# ===================================================================== Stage 3: scientific self-test
def _run_selftest(build_fn, spec_payload):
    """Runs INSIDE the isolated child. Returns {'passed': bool, 'cases': [{name, ok, reason}]}. Each check
    rejects with a specific reason. Never raises (wraps every probe)."""
    role = spec_payload["role"]
    nc = max(2, int(spec_payload["n_classes"]))
    Xtr, ytr, Xte, _ = _make_synth(spec_payload, seed=0)
    cases = []

    def _case(name, ok, reason="ok"):
        cases.append({"name": name, "ok": bool(ok), "reason": str(reason)})
        return bool(ok)

    all_ok = True

    # (a) VALID PREDICTIONS on a disjoint holdout (also covers (d) train-only fit).
    try:
        est = _AuthoredWrapper(build_fn(0), role)
        est.fit(Xtr, ytr)
        pred = np.asarray(est.predict(Xte))
        ok_len = pred.shape[0] == Xte.shape[0]
        ok_fin = bool(np.all(np.isfinite(pred.astype(float))))
        ok = ok_len and ok_fin
        reason = "ok" if ok else (f"length {pred.shape[0]} != {Xte.shape[0]}" if not ok_len else "non-finite predictions")
        all_ok &= _case("valid_predictions", ok, reason)
        if ok and role in ("classifier",):
            ints = np.allclose(pred, np.round(pred))
            inrange = pred.min() >= 0 and pred.max() < nc
            all_ok &= _case("label_range", ints and inrange,
                             "ok" if (ints and inrange) else f"labels not int in [0,{nc}): min={pred.min()} max={pred.max()}")
    except Exception as ex:  # noqa: BLE001
        all_ok &= _case("valid_predictions", False, f"{type(ex).__name__}: {str(ex)[:160]}")
        return {"passed": False, "cases": cases}

    # (b) DETERMINISM across two builds with the SAME seed -> byte-identical predictions.
    try:
        e1 = _AuthoredWrapper(build_fn(7), role); e1.fit(Xtr, ytr); p1 = np.asarray(e1.predict(Xte))
        e2 = _AuthoredWrapper(build_fn(7), role); e2.fit(Xtr, ytr); p2 = np.asarray(e2.predict(Xte))
        det = np.array_equal(p1, p2)
        all_ok &= _case("deterministic", det,
                        "ok" if det else "predictions differ across two runs with seed=7")
    except Exception as ex:  # noqa: BLE001
        all_ok &= _case("deterministic", False, f"{type(ex).__name__}: {str(ex)[:160]}")

    # (c) PERMUTATION-EQUIVARIANCE: predict(X[perm]) == predict(X)[perm]. A predictor that leaked test
    #     labels by row index (rather than computing from features) fails this.
    try:
        e3 = _AuthoredWrapper(build_fn(0), role); e3.fit(Xtr, ytr)
        base = np.asarray(e3.predict(Xte))
        rng = np.random.RandomState(123)
        perm = rng.permutation(Xte.shape[0])
        permuted = np.asarray(e3.predict(Xte[perm]))
        equi = np.array_equal(permuted, base[perm])
        all_ok &= _case("row_equivariant", equi,
                        "ok" if equi else "predictions not row-equivariant (possible index-based leakage)")
    except Exception as ex:  # noqa: BLE001
        all_ok &= _case("row_equivariant", False, f"{type(ex).__name__}: {str(ex)[:160]}")

    # (e) PARTIAL/ROBUST BUILD: a different seed still builds + fits (defaults robust).
    try:
        e4 = _AuthoredWrapper(build_fn(99), role); e4.fit(Xtr, ytr); e4.predict(Xte)
        all_ok &= _case("robust_build", True)
    except Exception as ex:  # noqa: BLE001
        all_ok &= _case("robust_build", False, f"{type(ex).__name__}: {str(ex)[:160]}")

    return {"passed": all_ok, "cases": cases}


def scientific_selftest(source, spec: EstimatorSpec):
    """Stage 3 wrapper: run the scientific self-test inside the isolated child. Returns (ok, report_dict).
    Static gate is assumed already passed by the caller (admit() runs it first). Never raises."""
    ok, result, reason = run_isolated(source, "selftest", spec.as_payload(),
                                      cpu_s=spec.cpu_s, mem_mb=spec.mem_mb, wall_s=spec.wall_s)
    if not ok:
        return False, {"passed": False, "cases": [{"name": "isolated_exec", "ok": False, "reason": reason}]}
    return bool(result.get("passed")), result


# ===================================================================== admit(): the three-stage gate
@dataclass
class AdmissionReport:
    admitted: bool
    reason: str
    static: dict
    selftest: Optional[dict] = None
    code: Optional[str] = None
    family: Optional[str] = None
    role: Optional[str] = None
    entrypoint: str = ENTRYPOINT
    code_digest: Optional[str] = None
    factory: Optional["AuthoredEstimator"] = None

    def as_dict(self):
        return {"admitted": self.admitted, "reason": self.reason, "static": self.static,
                "selftest": self.selftest, "family": self.family, "role": self.role,
                "code_digest": self.code_digest}


def admit(code, spec: EstimatorSpec, *, family=None, params=None, param_space=None,
          rationale="") -> AdmissionReport:
    """The frozen three-stage admission gate. admit(code, spec) -> AdmissionReport. On admitted=True the
    report carries .factory (an AuthoredEstimator) whose .catalog_entry() is shape-identical to a
    hand-written harness.CatalogEntry -- the EXISTING move_from_proposal turns it into a Move with zero loop
    changes. On rejection .admitted=False and .reason names the failing stage + the specific violation.
    Admission is SAFETY + BUILDABILITY + CONTRACT only -- NEVER a quality bar. Never raises."""
    # Stage 1: static AST gate (never exec ungated code).
    st = estimator_static_check(code, entrypoint=spec.entrypoint)
    if not st.ok:
        return AdmissionReport(False, "static gate: " + "; ".join(st.violations), st.as_dict(), code=code)

    # Stages 2+3: isolated execution + scientific self-test.
    ok, report = scientific_selftest(code, spec)
    if not ok:
        bad = [c for c in report.get("cases", []) if not c["ok"]]
        why = "; ".join(f"{c['name']}={c['reason']}" for c in bad) or "self-test failed"
        return AdmissionReport(False, "self-test: " + why, st.as_dict(), selftest=report, code=code)

    # admitted.
    try:
        from vectorforge import science
        cd = science.digest({"code": code, "role": spec.role})
    except Exception:  # noqa: BLE001 -- digest is for provenance only; never block admission on it
        import hashlib
        cd = "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()
    fam = f"{spec.family_prefix}__{_safe_family(family or 'method')}"
    ae = AuthoredEstimator(family=fam, role=spec.role, code=code,
                           default_params=dict(params or {}), param_space=dict(param_space or {}),
                           conformance=report, code_digest=cd, rationale=rationale,
                           entrypoint=spec.entrypoint)
    return AdmissionReport(True,
                           "passed static gate + isolated exec + scientific self-test; admitted as a candidate family",
                           st.as_dict(), selftest=report, code=code, family=fam, role=spec.role,
                           entrypoint=spec.entrypoint, code_digest=cd, factory=ae)


def _safe_family(name):
    return re.sub(r"[^a-z0-9_]", "", str(name).lower())[:40] or "method"


# ===================================================================== admitted method -> Move-compatible candidate
@dataclass
class AuthoredEstimator:
    """An admitted authored method. catalog_entry() is shape-identical to a hand-written harness.CatalogEntry,
    so the EXISTING move_from_proposal turns it into a Move with zero loop changes. worker_safe=False ALWAYS
    (the worker's build_model cannot reconstruct a novel class) -> runs only on the local in-process path,
    which is exactly the path that keeps the live estimator object and re-fits locally for certify."""
    family: str
    role: str
    code: str
    default_params: dict = field(default_factory=dict)
    param_space: dict = field(default_factory=dict)
    conformance: Optional[dict] = None
    code_digest: Optional[str] = None
    rationale: str = ""
    entrypoint: str = ENTRYPOINT

    def _builder(self):
        """Return ctor(clamped_params, seed) -> estimator. Re-execs the gated code ONCE per build in the
        restricted in-process namespace (the loop fits/predicts the LIVE object on the local in-process
        path, and re-fits the SAME ctor for the trusted certify; isolation is the ADMISSION gate). The seed
        is the build_estimator argument so a deterministic method consumes it directly."""
        code, role, entry = self.code, self.role, self.entrypoint

        def builder(clamped_params, seed):  # clamped_params accepted for catalog parity; method reads seed
            ok, build_fn, err = _compile_build(code, entry)
            if not ok:
                raise RuntimeError(f"authored build_estimator failed to compile: {err}")
            est = build_fn(int(seed))
            return _AuthoredWrapper(est, role)
        return builder

    def catalog_entry(self) -> CatalogEntry:
        """A harness.CatalogEntry whose builder is the authored build_estimator. params/grid come from
        param_space so the proposer can sweep the authored method exactly like a zoo family. prior_gain is
        modest (unproven); prior_cost conservative (unknown training cost)."""
        params, grid = {}, {}
        for name, vals in (self.param_space or {}).items():
            vlist = list(vals) if isinstance(vals, (list, tuple)) else [vals]
            if not vlist:
                continue
            if all(isinstance(x, numbers.Number) and not isinstance(x, bool) for x in vlist):
                lo, hi = float(min(vlist)), float(max(vlist))
                kind = "int" if all(float(x).is_integer() for x in vlist) else "float"
                params[name] = (kind, lo, hi)
            else:
                params[name] = ("choice", vlist)
            grid[name] = vlist
        b = self._builder()
        return CatalogEntry(
            family=self.family,
            builder=lambda p, s, _b=b: _b(p, s),
            params=params, grid=grid,
            prior_gain=0.04, prior_cost=2.0, worker_safe=False)   # local-only; never dispatched to a worker


def authored_moves(reports, catalog, *, tried=frozenset(), limit=4):
    """Turn admitted AuthoredEstimators into Moves using the EXISTING frozen harness.move_from_proposal.
    INTENDED INTEGRATION (done by hand later, NOT in this workflow): the loop registers each authored
    family's CatalogEntry into the per-task `cat` dict, then this helper emits grid Moves over it -- identical
    to how a zoo family is proposed. `reports` may be AdmissionReport objects or AuthoredEstimator objects."""
    moves = []
    for r in reports:
        ae = r.factory if isinstance(r, AdmissionReport) else r
        if ae is None:
            continue
        ce = ae.catalog_entry()
        catalog[ae.family] = ce            # register so resolve_family/move_from_proposal can build it
        for (fam, params) in _grid_configs(ce):
            m = move_from_proposal(catalog, fam, params, prefix="authored:")
            if m is not None and m.name not in tried:
                moves.append(m)
            if len(moves) >= limit:
                break
        if len(moves) >= limit:
            break
    return moves


# ===================================================================== the AUTHOR call (non-binding; DECLINE)
_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["family", "code", "rationale"],
    "properties": {
        "family": {"type": "string",
                   "description": "Short snake_case name for the method, e.g. 'gated_residual_forest'."},
        "code": {"type": "string",
                 "description": "Self-contained Python defining `def build_estimator(seed): ...` returning an "
                                "object with fit(X,y)/predict(X) (classifier/regressor) OR fit(X[,y])/"
                                "transform(X) (featurizer). Allowed imports ONLY: numpy, sklearn, math. No "
                                "os/sys/file/network/eval/exec/import-tricks/dunder-introspection. Thread the "
                                "seed into every random_state for determinism."},
        "params": {"type": "object", "additionalProperties": True,
                   "description": "Optional default hyperparameters (numeric/bool/str)."},
        "param_space": {"type": "object", "additionalProperties": True,
                        "description": "Optional {name: [grid values]} so the search can sweep the method."},
        "rationale": {"type": "string", "description": "One line: the inductive bias this method adds."},
    },
}

_SYSTEM = (
    "You are the AUTHOR node of an automated, certify-or-honest-fail model search. Write a NOVEL, "
    "self-contained sklearn-compatible estimator (or featurizer) as Python source defining "
    "`def build_estimator(seed)`. It returns a fresh object: a classifier/regressor exposes fit(X, y) and "
    "predict(X); a featurizer exposes fit(X) (or fit(X, y)) and transform(X). HARD CONSTRAINTS: import ONLY "
    "numpy, sklearn (not sklearn.datasets / sklearn.model_selection), math; no file/OS/network/subprocess; "
    "no eval/exec/compile/open/__import__; no dunder-attribute or introspection tricks. THREAD the seed into "
    "EVERY random_state so the method is deterministic. You are NON-BINDING: a frozen static gate + isolated "
    "sandbox + scientific self-test will reject anything unsafe or non-conforming, and only the frozen "
    "certifier decides whether the method is good. Aim for a genuinely different inductive bias from a plain "
    "forest/boosting/linear baseline (e.g. a learned feature interaction, a stacked/gated ensemble, a "
    "kernel/prototype construction). TO CLEAR THE GATE: do NOT subclass sklearn.base.BaseEstimator and do NOT "
    "import sklearn.base or sklearn.utils (their validation helpers use getattr, which is banned); instead "
    "COMPOSE existing sklearn estimators (e.g. Pipeline/FeatureUnion over StandardScaler, Nystroem, "
    "RandomForestClassifier, ExtraTreesClassifier, SVC, KNeighborsClassifier, LogisticRegression) or write a "
    "plain class with fit/predict using ONLY numpy and those allowed sklearn estimator classes. Keep it light "
    "enough to fit on a few hundred synthetic rows within a few CPU-seconds (no giant n_estimators, no nested "
    "cross-validation in fit).")


@dataclass
class AuthoringResult:
    authored: bool
    estimator: Optional[AuthoredEstimator]
    used_llm: bool
    reason: str
    admission: Optional[dict] = None
    error: Optional[str] = None
    usage: Optional[dict] = None


def author_estimator(spec: EstimatorSpec, *, api_key=None, use_llm=True, cache_path=None,
                     tenant_id="default", model=ops.DEFAULT_MODEL, timeout=120.0,
                     extra_context=None) -> AuthoringResult:
    """Author ONE novel estimator/featurizer for `spec` via the audited ops.llm_propose harness (replay
    cache, quarantine, deterministic fallback). kind='VERIFIABLE' (output is code that MUST clear the frozen
    three-stage admit() gate). There is NO deterministic substitute for inventing a method, so the fallback
    is an honest DECLINE -- the search keeps using the fixed zoo. No user data is sent (untrusted_inputs={}).
    `extra_context` (optional) carries ONLY trusted, instruction-side facts -- accurate environment versions
    and the author's OWN prior admission-failure reasons so it can revise (it NEVER carries task data, labels,
    or the sealed test, and it cannot relax the frozen gate -- admission still decides). Never raises."""
    tctx = {"role": spec.role, "n_features": spec.n_features, "n_classes": spec.n_classes,
            "allowed_imports": sorted(ALLOWED_IMPORT_ROOTS), "entrypoint": spec.entrypoint}
    if extra_context:
        tctx.update(extra_context)
    req = ops.LLMRequest(
        surface=SURFACE, system=_SYSTEM,
        trusted_context=tctx,
        untrusted_inputs={},
        model=model, schema=_SCHEMA, tool_name=_TOOL_NAME, prompt_version=PROMPT_VERSION,
        force_tool=True, max_tokens=40000)

    def _verify(raw):
        if not isinstance(raw, dict):
            return False, None, "not an object"
        code, fam = raw.get("code"), raw.get("family")
        if not isinstance(code, str) or not code.strip():
            return False, None, "no code emitted"
        if not isinstance(fam, str) or not fam.strip():
            return False, None, "no family name"
        rep = admit(code, spec, family=fam, params=raw.get("params"),
                    param_space=raw.get("param_space"), rationale=raw.get("rationale", ""))
        if not rep.admitted:
            return False, None, rep.reason
        return True, {"family": rep.family, "code": code, "params": raw.get("params") or {},
                      "param_space": raw.get("param_space") or {}, "selftest": rep.selftest,
                      "code_digest": rep.code_digest, "rationale": raw.get("rationale", "")}, None

    def _fallback(reason):
        return {"declined": True, "reason": reason}

    p = ops.llm_propose(req, kind="VERIFIABLE", verify=_verify, fallback=_fallback,
                        tenant_id=tenant_id, api_key=api_key, use_llm=use_llm,
                        cache_path=cache_path, timeout=timeout)
    val = p.value or {}
    if isinstance(val, dict) and val.get("declined"):
        return AuthoringResult(False, None, p.used_llm,
                               f"DECLINED: {val.get('reason', 'no conforming estimator')}",
                               error=p.error, usage=p.usage)
    est = AuthoredEstimator(family=val["family"], role=spec.role, code=val["code"],
                            default_params=val["params"], param_space=val["param_space"],
                            conformance=val.get("selftest"), code_digest=val.get("code_digest"),
                            rationale=val.get("rationale", ""), entrypoint=spec.entrypoint)
    return AuthoringResult(True, est, p.used_llm,
                           "passed static gate + isolated exec + scientific self-test; admitted as a candidate family",
                           admission=val.get("selftest"), usage=p.usage)
