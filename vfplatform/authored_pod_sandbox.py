"""Linux OS-isolated executor for LLM-authored code (RunPod pod; no namespaces/Docker/seccomp available).

Threat model + design (verified on the pod):
- Authored code runs as the unprivileged user `nobody` via `setpriv --reuid nobody --regid nogroup
  --clear-groups`, in a per-run scratch dir on the LOCAL overlay fs (/tmp) -- where Unix perms ARE enforced
  (the /workspace network volume IGNORES chmod, so secrets must never live there; they live root-600 in
  /root/.attestera, which `nobody` cannot read).
- The child env is SCRUBBED (env -i style): no ANTHROPIC_API_KEY / no secrets reach the child.
- rlimits (enforced on Linux): RLIMIT_AS (memory), RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_NPROC (fork-bomb guard);
  plus a parent wall-clock kill.
- ONLY train arrays (Xtr,ytr) + an eval-features matrix cross the boundary as .npy; the SEALED TEST is NEVER
  passed in. Predictions return as a JSON float list. The authored object never crosses the boundary.
- Residual (documented): network egress is not blocked (no seccomp/namespaces/iptables-owner on this pod).
  It is acceptable because there is nothing sensitive to exfil -- secrets are unreadable (local root-600),
  the env is scrubbed, /workspace is secret-free, and only non-secret train data is passed in. Add a seccomp
  socket-block when libseccomp is available to close it fully.

This module is invoked by the trusted parent (root). It does NOT promote anything; an admitted authored model
is just a candidate scored by the FROZEN certifier. NOT wired into loop.py until the adversarial pass is clean.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import pwd
import shutil
import stat as _stat
import subprocess
import tempfile
import time

import numpy as np

_RESULT_MAX_BYTES = 64 * 1024 * 1024   # result.json hard size cap (also bounded by child RLIMIT_FSIZE)


def _reap_uid(user, tries=6):
    """Loop SIGKILL-by-uid until no process of `user` remains (defeats fork-races that spawn a new child
    in the window between enumerating /proc and signalling). pkill rc 1 == nothing matched == done."""
    for _ in range(tries):
        rc = subprocess.run(["pkill", "-9", "-u", user], capture_output=True).returncode
        if rc != 0:
            return
        time.sleep(0.05)


def _purge_uid_files(uid):
    """Remove anything owned by this run's uid OUTSIDE the scratch dir (beacons in /tmp /var/tmp /dev/shm
    that rmtree(scratch) would miss). Safe because the uid is exclusively held for this run (flock slot)."""
    for d in ("/tmp", "/var/tmp", "/dev/shm"):
        subprocess.run(["find", d, "-xdev", "-uid", str(uid), "-delete"], capture_output=True)


def _read_result(rf, uid):
    """CONFUSED-DEPUTY-resistant root read of the child's result.json. Returns {ok,pred,reason}. Guards:
      O_NOFOLLOW   -> reject a child-planted symlink (root never follows it to a root-only path)
      O_NONBLOCK   -> open() never hangs if the child made result.json a FIFO/device
      S_ISREG      -> reject fifo/socket/device nodes
      st_uid==uid  -> must be owned by THIS run's sandbox uid (rejects a hardlink to a root-owned file)
      st_nlink==1  -> reject hardlink tricks (a 2nd link elsewhere to a sensitive inode)
      size cap     -> bound the parse (defuse decompression/quadratic-parse bombs)
      parse_constant -> reject Infinity/NaN JSON tokens
      finite vector -> pred must be a flat list of finite floats."""
    try:
        fd = os.open(rf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        return {"ok": False, "pred": None, "reason": f"result.json unreadable/symlink: {e}"}
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            return {"ok": False, "pred": None, "reason": "result.json not a regular file"}
        if st.st_uid != uid:
            return {"ok": False, "pred": None, "reason": f"result.json not owned by sandbox uid ({st.st_uid}!={uid})"}
        if st.st_nlink != 1:
            return {"ok": False, "pred": None, "reason": f"result.json has {st.st_nlink} hardlinks"}
        if st.st_size > _RESULT_MAX_BYTES:
            return {"ok": False, "pred": None, "reason": f"result.json too large ({st.st_size}B)"}
        with os.fdopen(fd) as fh:
            out = json.loads(fh.read(_RESULT_MAX_BYTES + 1),
                             parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"bad JSON const {c}")))
    except Exception as e:
        return {"ok": False, "pred": None, "reason": f"result.json parse: {e}"}
    pred = out.get("pred")
    if not isinstance(pred, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                             and math.isfinite(v) for v in pred):
        return {"ok": False, "pred": None, "reason": "pred not a finite float vector"}
    return {"ok": True, "pred": pred, "reason": None}

_SECRET_DIR = "/root/.attestera"   # local, root-600 -> unreadable by `nobody`

# Per-run dedicated uids (created on the pod: group `sbx` gid 60000, users sbx1..sbx8 uids 60001..60008,
# chosen INSIDE the userns uid_map range 0..65535 -- out-of-range uids fail setresuid with EINVAL).
# A dedicated uid per concurrent run lets us reap by uid (`pkill -9 -u sbxN`) to kill detached daemons /
# fork-bomb children that `setsid` away from the parent's process group, AND isolates concurrent runs from
# each other (a shared `nobody` would let one run ptrace/kill/read another's scratch).
_POOL = [f"sbx{i}" for i in range(1, 9)]
_POOL_GID = "sbx"
_LOCKDIR = "/tmp/.vfsbx_slots"


def _claim_slot(timeout=180):
    """Block until a pool uid is free; return (user, lockfile_handle). Hold the handle for the run's life."""
    os.makedirs(_LOCKDIR, exist_ok=True)
    try:
        os.chmod(_LOCKDIR, 0o700)
    except OSError:
        pass
    end = time.monotonic() + timeout
    while True:
        for u in _POOL:
            lf = open(os.path.join(_LOCKDIR, u + ".lock"), "w")
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return u, lf
            except BlockingIOError:
                lf.close()
        if time.monotonic() > end:
            raise RuntimeError("no free sandbox slot")
        time.sleep(0.2)


def _release_slot(lf):
    try:
        fcntl.flock(lf, fcntl.LOCK_UN)
    finally:
        lf.close()

_DRIVER = r'''
import json, os, resource
import numpy as np
mb = int(os.environ["VF_MEM_MB"]); cs = int(os.environ["VF_CPU_S"]); wd = os.environ["VF_WD"]
resource.setrlimit(resource.RLIMIT_AS, (mb * 1024 * 1024, mb * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, (cs, cs + 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))      # fork-bomb guard
except Exception:
    pass
src = open(os.path.join(wd, "source.py")).read()
g = {"__name__": "__authored__", "np": np}
exec(compile(src, "<authored>", "exec"), g)
build = g.get("build_estimator") or g.get("build")
if build is None:
    raise RuntimeError("authored code defines no build_estimator()/build()")
Xtr = np.load(os.path.join(wd, "Xtr.npy")); ytr = np.load(os.path.join(wd, "ytr.npy"))
Xev = np.load(os.path.join(wd, "Xev.npy"))
est = build(0); est.fit(Xtr, ytr); pred = est.predict(Xev)
json.dump({"pred": [float(p) for p in np.asarray(pred).reshape(-1)]},
          open(os.path.join(wd, "result.json"), "w"))
'''


def run_authored(source, Xtr, ytr, Xeval, *, cpu_s=20, mem_mb=4096, wall_s=60, python="python3", user=None):
    """Run authored `source` (defining build_estimator(seed)->estimator) as a dedicated pool uid on local fs,
    fit on (Xtr,ytr), predict on Xeval, return {ok, pred, reason}. Never raises; never leaks the authored
    object. Reaps ALL descendants of the run's uid on every exit path (incl. timeout) before cleanup."""
    slot_lf = None
    if user is None:
        user, slot_lf = _claim_slot()
    uid = pwd.getpwnam(user).pw_uid
    wd = tempfile.mkdtemp(dir="/tmp")          # LOCAL overlay fs (perms enforced; NOT /workspace)
    try:
        os.chmod(wd, 0o700)
        np.save(os.path.join(wd, "Xtr.npy"), np.asarray(Xtr, dtype=np.float32))
        np.save(os.path.join(wd, "ytr.npy"), np.asarray(ytr))
        np.save(os.path.join(wd, "Xev.npy"), np.asarray(Xeval, dtype=np.float32))
        with open(os.path.join(wd, "source.py"), "w") as fh:
            fh.write(source)
        with open(os.path.join(wd, "driver.py"), "w") as fh:
            fh.write(_DRIVER)
        subprocess.run(["chown", "-R", f"{user}:{_POOL_GID}", wd], check=True)   # child uid can read inputs/write result
        env = {"PATH": "/usr/bin:/bin", "HOME": wd, "TMPDIR": wd, "VF_WD": wd,
               "VF_MEM_MB": str(int(mem_mb)), "VF_CPU_S": str(int(cpu_s)),
               # SCRUBBED: no secret env vars. Thread/arena caps are mandatory on this 256-core box:
               # without them numpy/scipy/OpenBLAS spawn one thread + malloc arena PER CORE at import,
               # which is pathologically slow (burns RLIMIT_CPU before the child can run) and reserves
               # huge virtual address space. They also keep the sandboxed child off all 256 cores.
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2"}
        # --no-new-privs: even if the child execs a SUID-root binary (or a PATH-hijacked one the parent later
        # calls), the kernel will NOT grant new privileges -> kills the SUID/PATH-hijack escalation class.
        # --inh-caps/--ambient-caps cleared: no capability can be inherited or made ambient across the uid drop.
        cmd = ["setpriv", "--reuid", user, "--regid", _POOL_GID, "--clear-groups",
               "--no-new-privs", "--inh-caps=-all", "--ambient-caps=-all",
               python, os.path.join(wd, "driver.py")]
        try:
            p = subprocess.run(cmd, env=env, cwd=wd, capture_output=True, timeout=wall_s, text=True)
            timed_out = False
        except subprocess.TimeoutExpired as e:
            p, timed_out = e, True
        # Reap EVERY process owned by this run's uid (detached daemons / fork-bomb children that setsid away
        # from the parent's process group), looping to defeat fork-races, before reading results.
        _reap_uid(user)
        if timed_out:
            return {"ok": False, "pred": None, "reason": f"wall timeout {wall_s}s"}
        rf = os.path.join(wd, "result.json")
        if p.returncode != 0:
            return {"ok": False, "pred": None, "reason": f"child exit {p.returncode}: {(p.stderr or '')[-300:]}"}
        return _read_result(rf, uid)
    finally:
        _reap_uid(user)                 # belt-and-suspenders reap-loop
        _purge_uid_files(uid)           # remove out-of-scratch beacons owned by this uid
        shutil.rmtree(wd, ignore_errors=True)
        if slot_lf is not None:
            _release_slot(slot_lf)
