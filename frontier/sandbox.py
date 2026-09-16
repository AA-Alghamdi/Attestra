"""Real out-of-process sandbox for executing untrusted candidate code.

Why a subprocess and not in-process exec:
  The audit found PR18/PR19 running LLM-authored code via `exec(code, {"__builtins__":
  __builtins__})` in the orchestrator's own process. That is not a sandbox: the code shares
  the parent's memory, can exhaust it, can run forever, and a crash takes the loop down.

Here each Program runs in a separate OS process with:
  - a CPU-seconds rlimit (RLIMIT_CPU) and a best-effort address-space cap (RLIMIT_AS),
  - a wall-clock timeout enforced by the parent (kills the whole process group),
  - its own session so child threads/processes die with it.

The firewall: the child returns ONLY predictions (written to a .npy) plus a status line.
It never computes or returns a metric. The trusted parent scores and certifies.

Phase-0 limitation (documented, not hidden): this provides resource + crash isolation but
not network isolation. A production deployment runs this same runner inside a no-network
container or a Prime Intellect pod (the providers already exist in vfplatform/). The runner
contract is identical, so that is a substrate swap, not a rewrite.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from .program import Program, RunResult

# A fixed, trusted runner. It loads arrays + the candidate code, builds the estimator,
# fits on train, predicts on the eval split, and writes predictions out. Status on stdout.
_RUNNER = r'''
import sys, numpy as np
job, codepath, outpath, kind = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    d = np.load(job, allow_pickle=True)
    Xtr, ytr, Xev = d["Xtr"], d["ytr"], d["Xev"]
    code = open(codepath, "r").read()
except Exception as e:
    print("ERR:other:could not load job: %s" % (str(e)[:160],)); sys.exit(0)

ns = {}
try:
    exec(compile(code, "<candidate>", "exec"), ns)
except Exception as e:
    print("ERR:build:compile/exec failed: %s" % (str(e)[:160],)); sys.exit(0)
if "build_estimator" not in ns or not callable(ns["build_estimator"]):
    print("ERR:build:no callable build_estimator()"); sys.exit(0)
try:
    est = ns["build_estimator"]()
except Exception as e:
    print("ERR:build:build_estimator() raised: %s" % (str(e)[:160],)); sys.exit(0)

try:
    ytr2 = ytr.astype(str) if kind == "classification" else ytr.astype(float)
    est.fit(Xtr, ytr2)
    preds = est.predict(Xev)
    np.save(outpath, np.asarray(preds, dtype=object), allow_pickle=True)
    print("OK")
except MemoryError:
    print("ERR:oom:MemoryError during fit/predict")
except ImportError as e:
    print("ERR:import:%s" % (str(e)[:160],))
except Exception as e:
    print("ERR:fit:%s: %s" % (type(e).__name__, str(e)[:160]))
'''


def _preexec(cpu_seconds: int, address_mb: int):
    """Set resource limits in the child before exec (POSIX only)."""
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (ValueError, OSError):
        pass
    # RLIMIT_AS is unreliable on macOS (can break interpreter startup); best-effort only.
    if address_mb and sys.platform != "darwin":
        try:
            soft = address_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            pass


def run_program(program: Program, X_train, y_train, X_eval, *, kind: str,
                wall_seconds: float = 60.0, cpu_seconds: int = 55,
                address_mb: int = 4096) -> RunResult:
    """Fit `program` on (X_train, y_train) in an isolated subprocess and predict X_eval.

    Returns a RunResult carrying predictions (parent scores them) or a typed error.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="frontier_sbx_") as d:
        job = os.path.join(d, "job.npz")
        codep = os.path.join(d, "candidate.py")
        runp = os.path.join(d, "runner.py")
        outp = os.path.join(d, "preds.npy")

        np.savez(job,
                 Xtr=np.asarray(X_train, dtype=float),
                 ytr=np.asarray(y_train, dtype=object),
                 Xev=np.asarray(X_eval, dtype=float))
        with open(codep, "w") as f:
            f.write(program.code)
        with open(runp, "w") as f:
            f.write(_RUNNER)

        posix = os.name == "posix"
        try:
            proc = subprocess.Popen(
                [sys.executable, runp, job, codep, outp, kind],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=posix,
                preexec_fn=(lambda: _preexec(cpu_seconds, address_mb)) if posix else None,
            )
        except Exception as e:
            return RunResult(program.id, ok=False, error=f"spawn failed: {e}",
                             error_kind="other", wall_seconds=time.time() - t0)

        try:
            out, err = proc.communicate(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            if posix:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.communicate()
            return RunResult(program.id, ok=False, error=f"wall timeout {wall_seconds}s",
                             error_kind="timeout", wall_seconds=time.time() - t0)

        wall = time.time() - t0
        status = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
        if status == "OK" and os.path.exists(outp):
            preds = np.load(outp, allow_pickle=True)
            return RunResult(program.id, ok=True, preds=list(preds), wall_seconds=wall)

        if status.startswith("ERR:"):
            # format is "ERR:<kind>:<message>"; message may itself contain colons.
            parts = status.split(":", 2)
            error_kind = parts[1].strip() if len(parts) > 1 else "other"
            message = parts[2].strip() if len(parts) > 2 else status
            return RunResult(program.id, ok=False, error=message,
                             error_kind=error_kind, wall_seconds=wall)

        # No status / killed by rlimit (e.g. RLIMIT_CPU) -> stderr tail.
        tail = (err or "").strip().splitlines()[-1] if (err or "").strip() else "no output"
        kind_guess = "cpu" if proc.returncode and proc.returncode < 0 else "other"
        return RunResult(program.id, ok=False, error=tail[:200],
                         error_kind=kind_guess, wall_seconds=wall)
