"""Process-isolated sandbox for LLM-generated solution code.

Adapts the proven spawn + rlimits + AST gate pattern from vfplatform/authoring.py
for the generative engine's `solve(X_train, y_train, X_test) -> predictions` signature.

Security model:
  1. AST STATIC GATE: allowlist (numpy/sklearn/scipy/math), denylist (os/sys/subprocess/io/etc),
     banned attrs (file I/O, frame introspection), entrypoint check.
  2. PROCESS ISOLATION: multiprocessing.spawn -> fresh interpreter, no inherited modules/fds/sealed test.
     rlimits: RLIMIT_CPU (enforced, portable), RLIMIT_AS/RLIMIT_DATA (enforced on Linux, best-effort macOS).
     Parent wall-clock timeout via Pipe.poll() + terminate/kill: universal backstop.
  3. RESTRICTED BUILTINS: safe subset of builtins, allowlisted __import__, no __builtins__ access.
  4. SECRET SCRUBBING: env vars containing KEY/TOKEN/SECRET stripped in child.

Data is passed to the child via the pipe (pickled numpy arrays). Predictions return the same way.
"""
from __future__ import annotations

import ast
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────── AST gate config

ALLOWED_IMPORT_ROOTS = {"numpy", "np", "scipy", "sklearn", "math"}

DENIED_SCIPY_SUBMODULES = {"scipy.io"}  # file I/O bypass risk

ALLOWED_SKLEARN_SUBMODULES = {
    "sklearn", "sklearn.linear_model", "sklearn.ensemble", "sklearn.svm", "sklearn.tree",
    "sklearn.neighbors", "sklearn.preprocessing", "sklearn.base", "sklearn.utils",
    "sklearn.metrics", "sklearn.kernel_approximation", "sklearn.decomposition",
    "sklearn.pipeline", "sklearn.multiclass", "sklearn.naive_bayes", "sklearn.cluster",
    "sklearn.gaussian_process", "sklearn.neural_network", "sklearn.discriminant_analysis",
    "sklearn.compose", "sklearn.feature_selection", "sklearn.impute",
    "sklearn.kernel_ridge", "sklearn.isotonic", "sklearn.calibration",
    "sklearn.semi_supervised", "sklearn.cross_decomposition",
    "sklearn.model_selection",
}

DENIED_SKLEARN_SUBMODULES = {"sklearn.datasets", "sklearn.externals"}
# Note: sklearn.model_selection is ALLOWED because solution code may use cross_val_score
# for internal CV. The sealed test is never passed to solution code — the engine controls the split.

BANNED_NAMES = {
    "eval", "exec", "compile", "execfile", "input", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "type", "super", "classmethod",
    "staticmethod", "property", "breakpoint", "exit", "quit", "help", "copyright",
    "credits", "license", "open", "memoryview",
}

BANNED_ATTRS = {
    "__class__", "__subclasses__", "__bases__", "__mro__", "__dict__", "__globals__",
    "__code__", "__func__", "__self__", "__module__", "__qualname__", "__wrapped__",
    "__closure__", "__annotations__", "__kwdefaults__", "__defaults__",
}

DENYLIST_IDENTIFIERS = {
    "os", "sys", "subprocess", "socket", "builtins", "importlib", "ctypes", "pickle",
    "marshal", "shutil", "pathlib", "requests", "urllib", "http", "io", "tempfile",
    "threading", "asyncio", "ftplib", "smtplib", "signal", "resource", "multiprocessing",
    "__builtins__",
}

DENIED_RUNTIME_ATTRS = {
    "savetxt", "save", "savez", "savez_compressed", "load", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "ctypeslib", "fromregex", "DataSource", "frombuffer",
    "getbuffer", "fromiter", "f_back", "f_globals", "f_locals", "f_builtins", "f_code",
    "gi_frame", "gi_code", "cr_frame",
    # scipy.io file operations
    "savemat", "loadmat", "whosmat", "readsav", "netcdf_file", "wavfile",
    "arff", "mmread", "mmwrite", "mminfo", "hb_read", "hb_write",
}

_MAX_CODE_CHARS = 12000
_MAX_AST_NODES = 6000

# Default resource limits
_DEF_CPU_S = 90       # generous for sklearn stacking
_DEF_MEM_MB = 4096    # 4GB — sklearn uses joblib threading which needs virtual memory headroom
_DEF_WALL_S = 120.0   # 2min wall clock


# ─────────────────────────────────────────── AST static gate

@dataclass
class StaticCheckReport:
    ok: bool
    violations: List[str] = field(default_factory=list)
    has_entrypoint: bool = False


def static_check(code: str, entrypoint: str = "solve") -> StaticCheckReport:
    """AST-level allowlist/denylist for solution code. Never executes anything."""
    if not isinstance(code, str) or not code.strip():
        return StaticCheckReport(False, ["empty code"])
    if len(code) > _MAX_CODE_CHARS:
        return StaticCheckReport(False, [f"code exceeds {_MAX_CODE_CHARS} chars"])

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StaticCheckReport(False, [f"syntax error: {e}"])

    violations: List[str] = []
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > _MAX_AST_NODES:
        return StaticCheckReport(False, [f"AST too large: {n_nodes} nodes > {_MAX_AST_NODES}"])

    has_entrypoint = not entrypoint  # empty entrypoint = flat code, no function required

    for node in ast.walk(tree):
        # Check function definitions
        if entrypoint and isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            has_entrypoint = True
            n_args = len(node.args.args)
            if n_args < 3:
                violations.append(
                    f"{entrypoint}() must accept at least 3 args (X_train, y_train, X_test), got {n_args}"
                )

        # Banned names
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            violations.append(f"banned name: {node.id}")
        if isinstance(node, ast.Name) and node.id in DENYLIST_IDENTIFIERS:
            violations.append(f"denied identifier: {node.id}")

        # Banned attributes
        if isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                violations.append(f"banned attr: .{node.attr}")
            if node.attr in DENIED_RUNTIME_ATTRS:
                violations.append(f"denied runtime attr: .{node.attr}")

        # Import checks
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append(f"import not allowed: {name}")
                if root == "sklearn" and name in DENIED_SKLEARN_SUBMODULES:
                    violations.append(f"denied sklearn submodule: {name}")
                if root == "scipy" and name in DENIED_SCIPY_SUBMODULES:
                    violations.append(f"denied scipy submodule: {name} (file I/O risk)")

        # Infinite loop guard
        if isinstance(node, ast.While):
            if _is_constant_true(node.test) and not _has_break(node):
                violations.append("constant-true while without break (infinite loop)")

        # No global/nonlocal
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"global/nonlocal not permitted")

    if not has_entrypoint:
        violations.append(f"missing entrypoint: def {entrypoint}(X_train, y_train, X_test)")

    return StaticCheckReport(
        ok=len(violations) == 0,
        violations=violations,
        has_entrypoint=has_entrypoint,
    )


def _is_constant_true(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.Name) and node.id == "True":
        return True
    return False


def _has_break(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Break):
            return True
    return False


# ─────────────────────────────────────────── restricted builtins

_SAFE_BUILTINS = None


def _build_safe_builtins() -> Dict[str, Any]:
    import builtins as _b
    keep = (
        "len", "range", "enumerate", "str", "int", "float", "bool", "list", "dict", "tuple",
        "set", "frozenset", "sorted", "reversed", "sum", "min", "max", "abs", "round", "zip",
        "map", "filter", "any", "all", "isinstance", "issubclass", "repr", "ord", "chr", "slice",
        "next", "iter", "divmod", "pow", "hash", "id", "print", "format", "complex",
        "bytearray", "bytes", "callable", "classmethod", "staticmethod", "property", "super", "type",
        "ValueError", "KeyError", "IndexError", "TypeError", "ZeroDivisionError",
        "ArithmeticError", "Exception", "StopIteration", "RuntimeError", "AttributeError",
        "NotImplementedError", "OverflowError", "FloatingPointError", "ImportError",
        "MemoryError", "RecursionError",
    )
    safe = {k: getattr(_b, k) for k in keep if hasattr(_b, k)}
    safe["__build_class__"] = _b.__build_class__

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        full = str(name)
        root = full.split(".")[0]
        if level != 0:
            raise ImportError("relative import not permitted in sandbox")
        if root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"import of {full!r} not permitted in sandbox")
        if root == "sklearn" and full in DENIED_SKLEARN_SUBMODULES:
            raise ImportError(f"import of {full!r} denied (data/split access)")
        return __import__(name, globals, locals, fromlist, level)

    safe["__import__"] = _safe_import
    return safe


# ─────────────────────────────────────────── child process

def _child_main(conn, code: str, X_train, y_train, X_test, cpu_s: int, mem_mb: int):
    """Spawn target. Fresh interpreter, rlimits, restricted builtins.
    Runs solve(X_train, y_train, X_test) and sends predictions back via pipe."""
    try:
        # ---- rlimits
        try:
            import signal as _signal
            import resource as _resource

            def _on_xcpu(signum, frame):
                try:
                    conn.send(("error", f"cpu timeout ({cpu_s}s)"))
                    conn.close()
                finally:
                    os._exit(1)

            _signal.signal(_signal.SIGXCPU, _on_xcpu)
            _resource.setrlimit(_resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 2))
        except (ValueError, OSError, AttributeError):
            pass

        try:
            import resource as _resource
            nbytes = int(mem_mb) * 1024 * 1024
            for lim in ("RLIMIT_AS", "RLIMIT_DATA"):
                try:
                    r = getattr(_resource, lim)
                    soft, hard = _resource.getrlimit(r)
                    newhard = hard if hard != _resource.RLIM_INFINITY and hard < nbytes else nbytes
                    _resource.setrlimit(r, (nbytes, newhard))
                except (ValueError, OSError, AttributeError):
                    pass
        except ImportError:
            pass

        # ---- scrub secrets from env
        try:
            for k in list(os.environ.keys()):
                if "KEY" in k.upper() or "TOKEN" in k.upper() or "SECRET" in k.upper():
                    del os.environ[k]
        except Exception:
            pass

        # ---- compile and execute
        global _SAFE_BUILTINS
        if _SAFE_BUILTINS is None:
            _SAFE_BUILTINS = _build_safe_builtins()

        ns = {"__builtins__": _SAFE_BUILTINS, "__name__": "sandbox_solution"}
        try:
            exec(compile(code, "<sandbox_solution>", "exec"), ns)  # noqa: S102
        except Exception as ex:
            conn.send(("error", f"compile/exec: {type(ex).__name__}: {str(ex)[:300]}"))
            conn.close()
            return

        solve_fn = ns.get("solve") or ns.get("build_estimator")
        if not callable(solve_fn):
            conn.send(("error", "no callable solve() or build_estimator() found"))
            conn.close()
            return

        # ---- run the solution
        import inspect
        params = list(inspect.signature(solve_fn).parameters.keys())
        if len(params) >= 3:
            predictions = solve_fn(X_train, y_train, X_test)
        else:
            # build_estimator(seed) pattern
            est = solve_fn(42)
            est.fit(X_train, y_train)
            predictions = est.predict(X_test)

        predictions = np.asarray(predictions)

        # Sanity: finite, right length
        if predictions.shape[0] != X_test.shape[0]:
            conn.send(("error", f"predictions length {predictions.shape[0]} != X_test length {X_test.shape[0]}"))
            conn.close()
            return
        if not np.all(np.isfinite(predictions.astype(float))):
            conn.send(("error", "predictions contain non-finite values"))
            conn.close()
            return

        conn.send(("ok", predictions.tolist()))
        conn.close()

    except MemoryError:
        try:
            conn.send(("error", "MemoryError: memory cap exceeded"))
            conn.close()
        except Exception:
            pass
        os._exit(1)
    except Exception as ex:
        try:
            conn.send(("error", f"{type(ex).__name__}: {str(ex)[:300]}"))
            conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────── public API

@dataclass
class SandboxResult:
    """Result from sandboxed code execution."""
    ok: bool
    predictions: Optional[np.ndarray] = None
    error: str = ""
    static_violations: List[str] = field(default_factory=list)


def run_sandboxed(
    code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    cpu_s: int = _DEF_CPU_S,
    mem_mb: int = _DEF_MEM_MB,
    wall_s: float = _DEF_WALL_S,
    skip_static_check: bool = False,
) -> SandboxResult:
    """Execute solution code in a process-isolated sandbox.

    Steps:
      1. AST static gate (unless skip_static_check=True)
      2. Spawn child process with rlimits
      3. Execute solve(X_train, y_train, X_test) in child
      4. Return predictions via pipe

    Returns SandboxResult with ok=True and predictions, or ok=False with error.
    Never raises.
    """
    # Stage 1: static check
    if not skip_static_check:
        report = static_check(code)
        if not report.ok:
            return SandboxResult(
                ok=False,
                error=f"static check failed: {'; '.join(report.violations[:5])}",
                static_violations=report.violations,
            )

    # Stage 2: spawn isolated child
    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = multiprocessing.get_context()

    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Ensure arrays are numpy
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    p = ctx.Process(
        target=_child_main,
        args=(child_conn, code, X_train, y_train, X_test, cpu_s, mem_mb),
    )
    p.start()
    child_conn.close()

    # Wall-clock timeout: universal backstop
    got = parent_conn.poll(wall_s)
    if not got:
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
            p.join(2)
        parent_conn.close()
        return SandboxResult(ok=False, error=f"wall timeout ({wall_s}s)")

    try:
        status, payload = parent_conn.recv()
    except EOFError:
        parent_conn.close()
        p.join(2)
        return SandboxResult(ok=False, error=f"child died (rlimit kill / crash; exit={p.exitcode})")

    parent_conn.close()
    p.join(2)
    if p.is_alive():
        p.kill()
        p.join(2)

    if status == "ok":
        predictions = np.asarray(payload)
        return SandboxResult(ok=True, predictions=predictions)
    else:
        return SandboxResult(ok=False, error=str(payload))
