"""OS-level isolation runner for LLM-authored numeric code (Tier-2 hardening).

WHY THIS EXISTS
---------------
`vfplatform/authoring.py` lets the LLM author sklearn-compatible estimator/featurizer
source. Its original defense was a Python-level AST allowlist. An adversarial pass BROKE
that sandbox: numpy is allowlisted, and numpy's C-extension file I/O
(`np.fromfile`/`np.genfromtxt`/`np.load`/`np.loadtxt`/`np.memmap`/`np.savetxt`) reads and
writes the filesystem WITHOUT ever touching a denied Python `open`/`os` identifier, so the
admission path exfiltrated `vectorforge/.anthropic_key`. The in-process fit/predict path also
had full filesystem + network access and (on macOS) no enforceable memory cap.

LESSON: safe execution of LLM-authored NUMERIC code needs OS-LEVEL isolation -- no
filesystem, no network, hard CPU + memory + wall caps -- for ALL execution (admission AND
every fit/predict), not a Python allowlist. A Python allowlist cannot constrain a C
extension; only the kernel can.

WHAT THIS MODULE PROVIDES
-------------------------
`run_in_sandbox(...)` executes authored `build_estimator` + its fit/predict in a FRESH child
process under as many OS-level controls as the host supports, and returns predictions across
a process boundary (JSON files inside a locked, per-run tmpdir). Controls, with the empirical
enforcement we MEASURED on this Darwin host (arm64, Python 3.11 x86_64 under Rosetta):

  Filesystem: macOS `sandbox-exec` profile. allow-default, then DENY read+write of the
              secret directory (vectorforge/) and DENY all writes except the locked tmpdir.
              MEASURED: blocks `np.fromfile`/`open` reads of .anthropic_key AND file writes
              (PermissionError, errno 1). This is the only layer that stops the C-extension
              exfil, since it operates below libpython.
  Network:    `sandbox-exec` `(deny network*)`. MEASURED: socket.create_connection blocked.
  CPU:        `resource.setrlimit(RLIMIT_CPU)`. MEASURED ENFORCED on Darwin (SIGXCPU, exit 152).
  File size:  `resource.setrlimit(RLIMIT_FSIZE)`. MEASURED ENFORCED on Darwin.
  Open fds:   `resource.setrlimit(RLIMIT_NOFILE)`. MEASURED ENFORCED on Darwin.
  Memory:     `resource.setrlimit(RLIMIT_AS/RLIMIT_DATA)` is MEASURED *NOT* enforced on Darwin
              (`setrlimit` itself fails: "current limit exceeds maximum limit"; a 512MB numpy
              alloc succeeds under a 256MB cap). We set it anyway (ENFORCED on Linux/RunPod)
              AND add a parent RSS-poll watchdog that SIGKILLs the child when its resident set
              crosses the cap. MEASURED: the watchdog kills a numpy mem-bomb (rc -9). The
              watchdog is coarse (kills shortly AFTER the breach, with up to one poll interval
              of overshoot), so set `mem_mb` below the true host ceiling.
  Wall clock: parent watchdog SIGKILLs on wall timeout (backstop for everything, incl. a child
              that ignores SIGXCPU or a sandbox-exec that is unavailable).

PORTABILITY
-----------
  - Darwin (dev): sandbox-exec present at /usr/bin/sandbox-exec; fs+net isolation real; CPU/
    FSIZE/NOFILE rlimits enforced; memory via the RSS watchdog (RLIMIT_AS unenforced).
  - Linux/RunPod (prod): sandbox-exec ABSENT. There, fs/net isolation should come from the
    container/namespace boundary (e.g. Docker `--network=none`, read-only rootfs, no secret
    bind-mount -- MEASURED on this host: host fs invisible, network blocked, `--memory` OOM
    kill enforced), and RLIMIT_AS/RLIMIT_DATA ARE enforced by the Linux kernel. This module
    degrades gracefully: if sandbox-exec is missing it still applies rlimits + the watchdog and
    reports `fs_isolation="rlimit-only (NO sandbox-exec; rely on container)"`.

This module is INTENTIONALLY standalone. It is NOT imported by loop.py and is NOT wired into
the certifier here. It is the isolation primitive that the admission/fit/predict paths should
call once it is reviewed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Default caps. Conservative: an authored sklearn estimator on the synth self-test fits in
# well under these. Callers override per task.
_DEF_CPU_S = 20          # RLIMIT_CPU soft seconds (ENFORCED on Darwin + Linux)
_DEF_WALL_S = 40.0       # parent wall-clock backstop (always enforced)
_DEF_MEM_MB = 1024       # address-space cap; RSS-watchdog cap on Darwin, RLIMIT_AS on Linux
_DEF_FSIZE_MB = 32       # RLIMIT_FSIZE (ENFORCED on Darwin); authored code should not write big files
_DEF_NOFILE = 256        # RLIMIT_NOFILE (ENFORCED on Darwin)
_RSS_POLL_S = 0.1        # watchdog poll interval

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# The secret directory the confirmed attack exfiltrated. Read+write of this subtree is denied
# by the sandbox profile regardless of allow-default. Kept as a module constant so reviewers
# can see exactly what is protected. Resolved to an absolute, symlink-free path at profile
# build time.
_SECRET_DIR = "/Users/abdullahalghamdi/vectorforge-product/vectorforge"

# The confirmed exfil read the key out of vectorforge/. But an IDENTICAL copy of the key lives
# at the repo ROOT (vectorforge-product/.anthropic_key), and the sealed-test files live across
# the whole repo -- none of which are under _SECRET_DIR. With an allow-default profile that
# denies only _SECRET_DIR, authored code trivially reads the root key copy (and any host file).
# So we deny-read the ENTIRE product repo subtree, not just vectorforge/.
_REPO_ROOT = "/Users/abdullahalghamdi/vectorforge-product"

# Additional secret-store subtrees and any file literally named ".anthropic_key" anywhere are
# also denied, so a copy placed outside the repo cannot be read either. (regex below.)
_DENY_READ_SUBPATHS = (
    _REPO_ROOT,                                   # both .anthropic_key copies + sealed tests + source
    "/Users/abdullahalghamdi/.ssh",
    "/Users/abdullahalghamdi/.aws",
    "/Users/abdullahalghamdi/.config",
    "/Users/abdullahalghamdi/.anthropic",
)


@dataclass
class SandboxResult:
    ok: bool
    predictions: Optional[list] = None      # list[float] returned across the process boundary
    reason: str = ""                         # failure / kill reason
    returncode: Optional[int] = None
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    mechanisms: dict = field(default_factory=dict)  # which OS controls were actually applied


# --------------------------------------------------------------------------------------------
# sandbox-exec profile generation (Darwin only)
# --------------------------------------------------------------------------------------------
def _sbpl(path: str) -> str:
    """Quote a path for an SBPL string literal (paths here are program-controlled, but escape
    backslashes/quotes defensively)."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def build_sandbox_profile(jail_real: str, secret_dir: str = _SECRET_DIR) -> str:
    """Return an SBPL (Sandbox Profile Language) profile string.

    Model: allow-default (so the Rosetta-translated x86 interpreter, dyld cache, stdlib and
    site-packages all load -- a deny-default profile crashes the interpreter on this host,
    MEASURED exit 134), then carve out the dangerous capabilities:
      * DENY read+write of the secret directory  -> stops the .anthropic_key exfil.
      * DENY all writes, then RE-ALLOW writes only inside the locked tmpdir -> stops
        exfil-to-disk and tampering with sealed-test files / source on disk.
      * DENY all network -> stops outbound exfil.

    `jail_real` MUST be the realpath (symlink-free) of the per-run tmpdir; on macOS /tmp is a
    symlink to /private/tmp and the sandbox matches the resolved path (MEASURED: writes to the
    /tmp alias are denied while the /private/tmp realpath is allowed).
    """
    jail_real = os.path.realpath(jail_real)
    # Build the read-deny subpath list: every secret subtree, resolved symlink-free, PLUS the
    # caller-supplied secret_dir (kept for back-compat). Dedupe while preserving order.
    deny_subpaths: list[str] = []
    for p in (secret_dir, *_DENY_READ_SUBPATHS):
        rp = os.path.realpath(p)
        if rp not in deny_subpaths:
            deny_subpaths.append(rp)
    read_deny_block = "\n".join(
        f'          (subpath "{_sbpl(p)}")' for p in deny_subpaths
    )
    return textwrap.dedent(
        f"""\
        (version 1)
        (allow default)
        ; --- deny reading or writing every secret subtree (repo root holds a 2nd key copy
        ;     plus all sealed-test files; vectorforge/ holds the key the confirmed attack read;
        ;     plus ssh/aws/config/anthropic credential stores) ---
        (deny file-read-data file-read* file-write*
{read_deny_block})
        ; --- belt-and-braces: deny reading ANY file basenamed .anthropic_key, wherever it is,
        ;     so a copy placed outside the denied subtrees still cannot be exfiltrated ---
        (deny file-read-data file-read*
          (regex #"/\\.anthropic_key$"))
        ; --- deny ALL filesystem writes, then re-allow ONLY the locked per-run tmpdir ---
        (deny file-write*)
        (allow file-write*
          (subpath "{_sbpl(jail_real)}")
          (subpath "/dev"))
        ; --- no network egress of any kind ---
        (deny network*)
        """
    )


# --------------------------------------------------------------------------------------------
# Child program: runs entirely inside the sandbox, talks to the parent via JSON files.
# It receives the authored source + payload from in.json (already inside the jail, written by
# the parent) and writes result.json. It NEVER inherits the parent's modules/fds/secrets
# because sandbox-exec launches a brand-new interpreter process.
# --------------------------------------------------------------------------------------------
_CHILD_PROGRAM = r'''
import json, os, sys, resource, signal

JAIL = sys.argv[1]
IN = os.path.join(JAIL, "in.json")
OUT = os.path.join(JAIL, "result.json")

def emit(obj):
    # atomic-ish write inside the jail (the only writable place)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, OUT)

def main():
    with open(IN) as f:
        job = json.load(f)
    cpu_s = int(job["cpu_s"]); mem_mb = int(job["mem_mb"])
    fsize_mb = int(job["fsize_mb"]); nofile = int(job["nofile"])

    # --- rlimits. SIGXCPU handler so a CPU-cap kill is reported, not silent. ---
    def _on_xcpu(signum, frame):
        try:
            emit({"ok": False, "reason": "cpu timeout (RLIMIT_CPU)"})
        finally:
            os._exit(152)
    try:
        signal.signal(signal.SIGXCPU, _on_xcpu)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 2))
    except Exception:
        pass
    for lim, val in (("RLIMIT_FSIZE", fsize_mb * 1024 * 1024),
                     ("RLIMIT_NOFILE", nofile)):
        try:
            r = getattr(resource, lim)
            soft, hard = resource.getrlimit(r)
            newhard = val if (hard == resource.RLIM_INFINITY or val < hard) else hard
            resource.setrlimit(r, (val, newhard))
        except Exception:
            pass
    # RLIMIT_AS/DATA: enforced on Linux, no-op-and-may-raise on Darwin (handled).
    nbytes = mem_mb * 1024 * 1024
    for lim in ("RLIMIT_AS", "RLIMIT_DATA"):
        try:
            r = getattr(resource, lim)
            soft, hard = resource.getrlimit(r)
            newhard = nbytes if (hard == resource.RLIM_INFINITY or nbytes < hard) else hard
            resource.setrlimit(r, (nbytes, newhard))
        except Exception:
            pass

    # --- scrub any secret-bearing env vars that might have slipped in ---
    for k in list(os.environ.keys()):
        if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD", "ANTHROPIC")):
            try: del os.environ[k]
            except Exception: pass

    import numpy as np

    # --- compile the authored source under restricted builtins (defense-in-depth; the OS
    #     sandbox is the real boundary). The authored module must define build_estimator(). ---
    src = job["source"]
    g = {"__name__": "authored", "__builtins__": __builtins__}
    try:
        code = compile(src, "<authored>", "exec")
        exec(code, g)
    except Exception as ex:
        emit({"ok": False, "reason": f"compile/exec error: {type(ex).__name__}: {str(ex)[:300]}"})
        return
    entry = job.get("entrypoint", "build_estimator")
    if entry not in g or not callable(g[entry]):
        emit({"ok": False, "reason": f"authored source does not define callable {entry!r}"})
        return

    # --- build, fit, predict. Data comes in the job payload (already inside the jail). ---
    try:
        Xtr = np.asarray(job["X_train"], dtype=float)
        ytr = np.asarray(job["y_train"], dtype=float)
        Xte = np.asarray(job["X_test"], dtype=float)
        est = g[entry]()
        est.fit(Xtr, ytr)
        pred = np.asarray(est.predict(Xte), dtype=float)
        finite = bool(np.all(np.isfinite(pred)))
        emit({"ok": True, "predictions": pred.ravel().tolist(),
              "n": int(pred.shape[0]) if pred.ndim else 1, "finite": finite})
    except MemoryError:
        emit({"ok": False, "reason": "MemoryError (RLIMIT_AS, Linux) "})
        os._exit(1)
    except Exception as ex:
        emit({"ok": False, "reason": f"fit/predict error: {type(ex).__name__}: {str(ex)[:300]}"})

if __name__ == "__main__":
    main()
'''


def _peak_rss_mb(pid: int) -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL).strip()
        return (int(out) / 1024.0) if out else 0.0
    except Exception:
        return 0.0


def _kill_tree(pid: int) -> None:
    for sig in (signal.SIGKILL,):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def run_in_sandbox(
    source: str,
    X_train,
    y_train,
    X_test,
    *,
    entrypoint: str = "build_estimator",
    python_exe: str = sys.executable,
    pythonpath: Optional[str] = None,
    cpu_s: int = _DEF_CPU_S,
    wall_s: float = _DEF_WALL_S,
    mem_mb: int = _DEF_MEM_MB,
    fsize_mb: int = _DEF_FSIZE_MB,
    nofile: int = _DEF_NOFILE,
    secret_dir: str = _SECRET_DIR,
) -> SandboxResult:
    """Execute authored `source` (which must define `entrypoint`, default build_estimator) and
    its estimator's fit/predict in an OS-isolated child. Returns predictions across a file
    boundary inside a locked, per-run tmpdir.

    Never raises for sandbox/child failures; returns SandboxResult(ok=False, reason=...).
    The arrays are passed as plain Python lists in in.json (no pickle -> no pickle RCE).
    """
    mechanisms: dict[str, str] = {}
    is_darwin = platform.system() == "Darwin"
    have_sbexec = is_darwin and os.path.exists(_SANDBOX_EXEC)

    # ---- locked per-run tmpdir: 0700, the ONLY writable location for the child ----
    jail = tempfile.mkdtemp(prefix="vf_authsbx_")
    try:
        os.chmod(jail, stat.S_IRWXU)  # 0700
        jail_real = os.path.realpath(jail)

        # serialize the job (arrays as lists -> no pickle on the boundary)
        try:
            import numpy as _np
            x_tr = _np.asarray(X_train, dtype=float).tolist()
            y_tr = _np.asarray(y_train, dtype=float).tolist()
            x_te = _np.asarray(X_test, dtype=float).tolist()
        except Exception as ex:  # noqa: BLE001
            return SandboxResult(ok=False, reason=f"input arrays not coercible: {ex}",
                                 mechanisms=mechanisms)
        job = {
            "source": source, "entrypoint": entrypoint,
            "cpu_s": int(cpu_s), "mem_mb": int(mem_mb),
            "fsize_mb": int(fsize_mb), "nofile": int(nofile),
            "X_train": x_tr, "y_train": y_tr, "X_test": x_te,
        }
        with open(os.path.join(jail_real, "in.json"), "w") as f:
            json.dump(job, f)

        child_py = os.path.join(jail_real, "child.py")
        with open(child_py, "w") as f:
            f.write(_CHILD_PROGRAM)

        # ---- assemble the command ----
        inner = [python_exe, "-I", "-B", child_py, jail_real]  # -I isolated, -B no .pyc
        if have_sbexec:
            profile = build_sandbox_profile(jail_real, secret_dir)
            prof_path = os.path.join(jail_real, "profile.sb")
            with open(prof_path, "w") as f:
                f.write(profile)
            cmd = [_SANDBOX_EXEC, "-f", prof_path] + inner
            mechanisms["filesystem"] = "sandbox-exec (deny secret dir + deny writes outside jail)"
            mechanisms["network"] = "sandbox-exec (deny network*)"
        else:
            cmd = inner
            mechanisms["filesystem"] = "rlimit-only (NO sandbox-exec; rely on container/namespace)"
            mechanisms["network"] = "NONE in-runner (rely on container --network=none)"

        # ---- launch in the jail, with a scrubbed environment ----
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": jail_real,
            "TMPDIR": jail_real,
            "PYTHONHASHSEED": "0",
        }
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        mechanisms["cpu"] = f"RLIMIT_CPU={cpu_s}s (enforced Darwin+Linux)"
        mechanisms["fsize"] = f"RLIMIT_FSIZE={fsize_mb}MB (enforced Darwin)"
        mechanisms["nofile"] = f"RLIMIT_NOFILE={nofile} (enforced Darwin)"
        mechanisms["memory"] = (
            f"RLIMIT_AS={mem_mb}MB (enforced Linux) + parent RSS-poll watchdog "
            f"(Darwin backstop; RLIMIT_AS NOT enforced there)"
        )
        mechanisms["wall"] = f"parent wall-clock kill at {wall_s}s"

        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, cwd=jail_real, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
        except Exception as ex:  # noqa: BLE001
            return SandboxResult(ok=False, reason=f"spawn failed: {ex}", mechanisms=mechanisms)

        # ---- parent watchdog: poll RSS (memory) and wall clock; SIGKILL on breach ----
        peak = 0.0
        killed_reason = None
        mem_cap_mb = float(mem_mb)
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            rss = _peak_rss_mb(proc.pid)
            if rss > peak:
                peak = rss
            if rss > mem_cap_mb:
                _kill_tree(proc.pid)
                killed_reason = f"memory cap exceeded (RSS {rss:.0f}MB > {mem_cap_mb:.0f}MB)"
                break
            if (time.time() - t0) > wall_s:
                _kill_tree(proc.pid)
                killed_reason = f"wall timeout ({wall_s}s)"
                break
            time.sleep(_RSS_POLL_S)

        try:
            out, err = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            _kill_tree(proc.pid)
            out, err = b"", b""
        wall = time.time() - t0
        rc = proc.returncode

        if killed_reason is not None:
            return SandboxResult(ok=False, reason=killed_reason, returncode=rc,
                                 wall_s=wall, peak_rss_mb=peak, mechanisms=mechanisms)

        # ---- read the result the child wrote into the jail ----
        res_path = os.path.join(jail_real, "result.json")
        if not os.path.exists(res_path):
            tail = (err or b"")[-400:].decode("utf-8", "replace")
            return SandboxResult(ok=False,
                                 reason=f"child produced no result (rc={rc}); stderr: {tail}",
                                 returncode=rc, wall_s=wall, peak_rss_mb=peak,
                                 mechanisms=mechanisms)
        try:
            with open(res_path) as f:
                payload = json.load(f)
        except Exception as ex:  # noqa: BLE001
            return SandboxResult(ok=False, reason=f"result.json unreadable: {ex}",
                                 returncode=rc, wall_s=wall, peak_rss_mb=peak,
                                 mechanisms=mechanisms)

        if not payload.get("ok"):
            return SandboxResult(ok=False, reason=payload.get("reason", "unknown child error"),
                                 returncode=rc, wall_s=wall, peak_rss_mb=peak,
                                 mechanisms=mechanisms)
        return SandboxResult(ok=True, predictions=payload.get("predictions"),
                             reason="", returncode=rc, wall_s=wall, peak_rss_mb=peak,
                             mechanisms=mechanisms)
    finally:
        shutil.rmtree(jail, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# Self-test: runs a benign authored estimator (must succeed) and four ATTACK estimators (must
# be contained). Mirrors the rigor rule: run the module's self-test before declaring it done.
# --------------------------------------------------------------------------------------------
_BENIGN = '''
import numpy as np
class _OLS:
    def fit(self, X, y):
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self
    def predict(self, X):
        return X @ self.beta_
def build_estimator():
    return _OLS()
'''

# Attack 1: numpy C-extension file read of the secret (the CONFIRMED exfil vector).
_ATK_FROMFILE = (
    "import numpy as np\n"
    f"KEY = {_SECRET_DIR + '/.anthropic_key'!r}\n"
    "class _Steal:\n"
    "    def fit(self, X, y):\n"
    "        self.leak_ = np.fromfile(KEY, dtype=np.uint8)  # must be BLOCKED by sandbox-exec\n"
    "        # encode the stolen bytes into predictions to detect any leak across the boundary\n"
    "        self.n_ = int(self.leak_.sum())\n"
    "        return self\n"
    "    def predict(self, X):\n"
    "        return np.full(X.shape[0], float(self.n_))\n"
    "def build_estimator():\n"
    "    return _Steal()\n"
)

# Attack 1b: read the SECOND key copy at the repo root (NOT under _SECRET_DIR). This is the
# escape an adversarial pass confirmed against the allow-default-deny-one-dir profile: the key
# is exfiltrated by encoding its bytes into the returned predictions. Must be BLOCKED now that
# the whole repo subtree (+ any */.anthropic_key) is read-denied.
_ATK_ROOTKEY = (
    "import numpy as np\n"
    f"KEY = {_REPO_ROOT + '/.anthropic_key'!r}\n"
    "class _Steal:\n"
    "    def fit(self, X, y):\n"
    "        with open(KEY, 'rb') as f:\n"
    "            self.data_ = f.read().strip()\n"
    "        return self\n"
    "    def predict(self, X):\n"
    "        return np.array([float(b) for b in self.data_], dtype=float)\n"
    "def build_estimator():\n"
    "    return _Steal()\n"
)

# Attack 2: exfil-to-disk write outside the jail.
_ATK_WRITE = '''
import numpy as np
class _Dump:
    def fit(self, X, y):
        np.savetxt("/tmp/vf_exfil_attempt.txt", X)  # must be BLOCKED (write outside jail)
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
def build_estimator():
    return _Dump()
'''

# Attack 3: network exfil.
_ATK_NET = '''
import numpy as np, socket
class _Phone:
    def fit(self, X, y):
        socket.create_connection(("1.1.1.1", 443), timeout=3)  # must be BLOCKED
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
def build_estimator():
    return _Phone()
'''

# Attack 4: memory bomb (RLIMIT_AS unenforced on Darwin -> RSS watchdog must kill).
_ATK_MEMBOMB = '''
import numpy as np, time
class _Bomb:
    def fit(self, X, y):
        self.chunks_ = []
        for _ in range(400):
            c = np.ones(64*1024*1024//8, dtype=np.float64)  # +64MB each
            c[0] = 1.0
            self.chunks_.append(c)
            time.sleep(0.02)
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
def build_estimator():
    return _Bomb()
'''

# Attack 5: CPU bomb (RLIMIT_CPU must kill).
_ATK_CPUBOMB = '''
class _Spin:
    def fit(self, X, y):
        x = 0
        while True:
            x += sum(i*i for i in range(100000))
        return self
    def predict(self, X):
        return [0.0]
def build_estimator():
    return _Spin()
'''


def _selftest() -> int:
    import numpy as np
    rng = np.random.RandomState(0)
    Xtr = rng.randn(60, 3); ytr = Xtr @ np.array([1.0, 2.0, -1.0]) + 0.05 * rng.randn(60)
    Xte = rng.randn(10, 3)

    print(f"host: {platform.system()} {platform.machine()}  "
          f"sandbox-exec: {os.path.exists(_SANDBOX_EXEC)}  python: {sys.executable}")
    fails = 0

    # benign must succeed and return predictions
    r = run_in_sandbox(_BENIGN, Xtr, ytr, Xte)
    ok_benign = r.ok and r.predictions is not None and len(r.predictions) == 10
    print(f"[benign  ] ok={r.ok} n_pred={None if r.predictions is None else len(r.predictions)} "
          f"wall={r.wall_s:.2f}s peakRSS={r.peak_rss_mb:.0f}MB  reason={r.reason!r}")
    print(f"           mechanisms={r.mechanisms}")
    if not ok_benign:
        print("  FAIL: benign estimator did not return predictions"); fails += 1

    # attacks must be CONTAINED (ok=False)
    for name, src, mem, cpu, wall in [
        ("fromfile", _ATK_FROMFILE, 1024, 20, 40),
        ("rootkey ", _ATK_ROOTKEY, 1024, 20, 40),
        ("write   ", _ATK_WRITE, 1024, 20, 40),
        ("network ", _ATK_NET, 1024, 20, 40),
        ("membomb ", _ATK_MEMBOMB, 256, 20, 40),   # tight mem cap so the watchdog fires fast
        ("cpubomb ", _ATK_CPUBOMB, 1024, 2, 40),    # tight CPU cap
    ]:
        r = run_in_sandbox(src, Xtr, ytr, Xte, mem_mb=mem, cpu_s=cpu, wall_s=wall)
        contained = (not r.ok)
        # for the secret-read attack, also verify no plaintext key leaked into the result
        leaked_secret = bool(r.predictions) and any(c > 0 for c in (r.predictions or []))
        verdict = "CONTAINED" if contained else "ESCAPED"
        print(f"[atk:{name}] {verdict:9s} ok={r.ok} rc={r.returncode} "
              f"wall={r.wall_s:.2f}s peakRSS={r.peak_rss_mb:.0f}MB reason={r.reason!r}")
        if not contained:
            print(f"  FAIL: attack {name.strip()} was NOT contained"); fails += 1

    print(f"\nself-test: {'PASS' if fails == 0 else f'FAIL ({fails})'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(_selftest())
