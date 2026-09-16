"""Tier-2 (LINUX_UID) sandbox substrate: real uid + network isolation via Linux namespaces.

This is the substrate the sandbox_policy contract (design 05 section 8 step 2) says lives in its own
module and registers itself with `register_substrate(Tier.LINUX_UID, ...)`. Until it loads, requesting
Tier-2/3 honestly degrades to Tier-1 (LOCAL, resource-only). Importing this module on a capable Linux
host upgrades untrusted execution from "resource-only" to "uid-confined + no-new-privs + network-isolated".

What it adds over the Phase-0 LOCAL runner (frontier/sandbox.py):
  - **uid confinement**: the candidate runs inside an unprivileged user namespace
    (`unshare --user --map-root-user`); it is root-inside / unprivileged-outside, so it cannot touch
    anything the launching user can't, and a kernel uid drop holds even if AST triage is bypassed.
  - **no-new-privs**: `setpriv --no-new-privs` blocks setuid/fcaps escalation for the whole subtree.
  - **network isolation**: a fresh network namespace (`unshare --net`) with only a down loopback ->
    the candidate has NO route to the network (the confirmed exfil channel), enforced by the kernel,
    not by an allow-list. (Probed live: socket.create_connection from inside raises OSError.)
  - everything Tier-1 already gives: RLIMIT_CPU + RLIMIT_AS (preexec), wall-clock backstop with a
    process-group kill, and the predictions-only firewall (the child returns ONLY a .npy of preds).

The runner signature is byte-for-byte the frozen `run_program` contract (invariant I3) so nothing above
the sandbox changes when the tier changes:

    run_program_t2(program, X_train, y_train, X_eval, *, kind,
                   wall_seconds, cpu_seconds, address_mb) -> RunResult

Honest degradation (never a false claim of isolation):
  - On a non-Linux host, or one without `unshare`/`setpriv`, or where an unprivileged user namespace
    cannot be created (some hardened kernels disable `kernel.unprivileged_userns_clone`), this module
    does NOT register itself. The policy then keeps Tier-2 unavailable and degrades, exactly as before.
  - The capability is *probed by actually creating the namespace* at import, not merely by checking that
    the binaries exist -- so a registered T2 is a T2 that demonstrably works on THIS host.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from ..program import Program, RunResult
from ..sandbox import _RUNNER, _preexec
from .sandbox_policy import Tier, register_substrate


def _userns_netns_works() -> bool:
    """Return True iff THIS host can actually create an unprivileged user+network namespace.

    We don't trust binary presence alone -- we run the real syscall path once. A no-op
    `unshare --user --map-root-user --net true` succeeds only when the kernel permits unprivileged
    userns creation (Debian/Ubuntu default-on; some hardened/distro kernels disable it). This makes a
    registered Tier-2 an *evidenced* Tier-2, not a wish.
    """
    if os.name != "posix" or not sys.platform.startswith("linux"):
        return False
    if not (shutil.which("unshare") and shutil.which("setpriv")):
        return False
    try:
        p = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--net", "--",
             "setpriv", "--no-new-privs", "--", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ns_prefix() -> list:
    """The namespace-entry argv prefix placed before the python runner.

    `--user --map-root-user` -> unprivileged user namespace (uid confinement).
    `--net`                  -> fresh network namespace, no routable interface (egress blocked).
    `setpriv --no-new-privs` -> no privilege escalation for the candidate subtree.
    """
    return [
        "unshare", "--user", "--map-root-user", "--net", "--",
        "setpriv", "--no-new-privs", "--",
    ]


def run_program_t2(program: Program, X_train, y_train, X_eval, *, kind: str,
                   wall_seconds: float = 60.0, cpu_seconds: int = 55,
                   address_mb: int = 4096) -> RunResult:
    """Tier-2 substrate: identical contract to frontier.sandbox.run_program, stronger isolation.

    Fits `program` on (X_train, y_train) inside a uid- and network-confined subprocess and predicts
    X_eval. Returns a RunResult carrying predictions only (the trusted parent scores + certifies); the
    firewall is preserved end to end. Resource limits, wall-clock kill, and error taxonomy mirror the
    Phase-0 runner so callers above the sandbox are unchanged.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="frontier_t2_") as d:
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

        argv = _ns_prefix() + [sys.executable, runp, job, codep, outp, kind]
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
                preexec_fn=lambda: _preexec(cpu_seconds, address_mb),
            )
        except Exception as e:
            return RunResult(program.id, ok=False, error=f"t2 spawn failed: {e}",
                             error_kind="other", wall_seconds=time.time() - t0)

        try:
            out, err = proc.communicate(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            return RunResult(program.id, ok=False, error=f"wall timeout {wall_seconds}s",
                             error_kind="timeout", wall_seconds=time.time() - t0)

        wall = time.time() - t0
        status = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
        if status == "OK" and os.path.exists(outp):
            preds = np.load(outp, allow_pickle=True)
            return RunResult(program.id, ok=True, preds=list(preds), wall_seconds=wall)

        if status.startswith("ERR:"):
            parts = status.split(":", 2)
            error_kind = parts[1].strip() if len(parts) > 1 else "other"
            message = parts[2].strip() if len(parts) > 2 else status
            return RunResult(program.id, ok=False, error=message,
                             error_kind=error_kind, wall_seconds=wall)

        tail = (err or "").strip().splitlines()[-1] if (err or "").strip() else "no output"
        kind_guess = "cpu" if proc.returncode and proc.returncode < 0 else "other"
        return RunResult(program.id, ok=False, error=tail[:200],
                         error_kind=kind_guess, wall_seconds=wall)


# Self-register at import IFF this host can actually create the namespaces (evidenced, not assumed).
T2_REGISTERED = _userns_netns_works()
if T2_REGISTERED:
    register_substrate(Tier.LINUX_UID, run_program_t2)


__all__ = ["run_program_t2", "T2_REGISTERED"]
