"""Restricted execution of statically-gated connector code (defense in depth, NOT a hardened boundary).

Runs the entrypoint with a safe-builtins-only namespace, only the allowlisted modules injected, and a
SIGALRM wall-clock timeout to bound runaway loops. Callers MUST run static_check first; this assumes
the code already passed the AST allowlist.
"""
import json
import re
import csv
import io
import math
import datetime
import signal
from contextlib import contextmanager

# A minimal safe builtins set: data ops only. No open/exec/eval/import/getattr.
_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ("len", "range", "enumerate", "str", "int", "float", "bool", "list", "dict", "tuple",
              "set", "frozenset", "sorted", "reversed", "sum", "min", "max", "abs", "round", "zip",
              "map", "filter", "any", "all", "isinstance", "repr", "format", "ord", "chr", "print",
              "ValueError", "KeyError", "IndexError", "TypeError", "Exception", "StopIteration",
              "slice", "next", "iter")
}
_ALLOWED_MODULES = {"json": json, "re": re, "csv": csv, "io": io, "math": math, "datetime": datetime}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """A restricted __import__ so that ALLOWLISTED `import` statements execute (the static gate already
    rejects non-allowlisted ones; this also blocks them at runtime as defense in depth)."""
    root = str(name).split(".")[0]
    if level == 0 and root in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[root]
    raise ImportError(f"import of {name!r} is not permitted in the sandbox")


_SAFE_BUILTINS["__import__"] = _safe_import


class SandboxError(Exception):
    pass


@contextmanager
def _time_limit(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"connector exceeded {seconds}s")
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def compile_function(code: str, entrypoint, *, timeout=3.0):
    """Exec statically-gated `code` once in a restricted namespace and return a timeout-wrapped handle to
    `entrypoint`, callable many times (each call bounded). Returns (ok, fn, error). Callers MUST run
    static_check first. Used by A7 to call build_attack/recover repeatedly without re-exec per row."""
    ns = {"__builtins__": _SAFE_BUILTINS}
    ns.update(_ALLOWED_MODULES)
    try:
        exec(compile(code, "<authored>", "exec"), ns)  # noqa: S102  statically-gated, restricted ns
    except Exception as ex:  # noqa: BLE001
        return False, None, f"exec error: {type(ex).__name__}: {str(ex)[:160]}"
    fn = ns.get(entrypoint)
    if not callable(fn):
        return False, None, f"entrypoint {entrypoint!r} not defined as a function"

    def _wrapped(*a, **k):
        with _time_limit(timeout):
            return fn(*a, **k)
    return True, _wrapped, None


def run_entrypoint(code: str, payload, *, entrypoint="load", timeout=3.0):
    """Exec `code` in a restricted namespace and call entrypoint(payload). Returns (ok, result, error).
    Never raises; any failure is reported as (False, None, msg)."""
    ns = {"__builtins__": _SAFE_BUILTINS}
    ns.update(_ALLOWED_MODULES)
    try:
        compiled = compile(code, "<authored_connector>", "exec")
        exec(compiled, ns)  # noqa: S102  intentional: statically-gated code, restricted namespace
    except Exception as ex:  # noqa: BLE001
        return False, None, f"load/exec error: {type(ex).__name__}: {str(ex)[:160]}"
    fn = ns.get(entrypoint)
    if not callable(fn):
        return False, None, f"entrypoint {entrypoint!r} not defined as a function"
    try:
        with _time_limit(timeout):
            result = fn(payload)
    except TimeoutError as ex:
        return False, None, str(ex)
    except Exception as ex:  # noqa: BLE001
        return False, None, f"runtime error: {type(ex).__name__}: {str(ex)[:160]}"
    return True, result, None
