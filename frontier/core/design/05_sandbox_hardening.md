# 05. Sandbox / safety hardening for the frontier autoresearcher

Status: design (implementation-ready). Scope: execution of arbitrary LLM-authored
model code (sklearn-compatible estimators today, neural nets / torch later) under a
sealed-test certify-or-decline discipline.

This document is deliberately honest about what is *enforced* at each tier and what is
only *advisory*. The governing rule: never log or claim a protection the active tier did
not actually enforce. Every claim below is annotated ENFORCED, BEST-EFFORT, or ADVISORY.

References read for this design:
- `frontier/sandbox.py` (Phase-0 spine: subprocess + RLIMIT_CPU + best-effort RLIMIT_AS +
  wall timeout + process-group kill; predictions-only firewall).
- `frontier/program.py`, `frontier/CONTRACT.md`, `frontier/ROADMAP.md` (the frozen
  `run_program` signature and the standing invariants the sandbox must preserve).
- `vfplatform/authored_pod_sandbox.py` (Linux pod: dedicated per-run uid via `setpriv`,
  `--no-new-privs`, scrubbed env, rlimits incl. RLIMIT_NPROC, uid-reap loop, confused-deputy
  result read). Read-only reference; this design does not import it.
- `vfplatform/authoring.py` (AST allow-list gate + spawn-not-fork isolation + the documented
  "AST allow-list is necessary-not-sufficient for numpy/C-extensions" caveat, and the
  confirmed numpy-I/O exfiltration finding). Read-only reference.

---

## 0. Design invariants (inherited from CONTRACT.md / ROADMAP.md; non-negotiable)

I1. **Predictions-only firewall.** Authored code returns predictions for a requested split.
    It never computes or returns a metric. Every decision number is computed by the trusted
    parent (`certify.py` / `vectorforge.science`). The sandbox boundary is the firewall.

I2. **Sealed test never crosses the boundary.** Authored code receives train features+labels
    plus the *features* of the eval split it must predict. It never receives the labels for
    any split it is scored on, and it never receives the sealed split's contents during
    selection. Scoring labels stay in the parent. (Detailed in section 6.)

I3. **The runner contract is identical across tiers.** Tier differences are *substrate* swaps
    underneath one fixed `run_program(program, X_train, y_train, X_eval, *, kind, wall_seconds,
    cpu_seconds, address_mb) -> RunResult`. No caller above the sandbox changes when the tier
    changes. This is already true of the Phase-0 contract in `CONTRACT.md` and must stay true.

I4. **Honest tiering.** The engine selects a tier by trust and availability, logs the *active*
    tier in every `RunResult`, and degrades by *announcing the weaker guarantee*, never by
    silently pretending the stronger one held.

---

## 1. Threat model -> defense map

"Untrusted authored code" = a Python string the LLM wrote, executed to produce predictions.
The adversary is (a) an accidentally-pathological program (infinite loop, OOM) and
(b) a worst case: a program written to exfiltrate the sealed test, the API keys, or to escape
the box. We design for (b); (a) falls out for free.

| # | What authored code could do | Concrete vector | Defense (tier where ENFORCED) |
|---|---|---|---|
| T1 | Exhaust **memory** | allocate a huge array; `np.ones((1e6,1e6))` | RLIMIT_AS/RLIMIT_DATA (T2/T3 ENFORCED; T1 BEST-EFFORT, unenforced on macOS) + container memory cgroup (T3) + wall-timeout backstop (all tiers ENFORCED) |
| T2 | Exhaust **CPU** / spin forever | `while True: pass`; `O(2^n)` fit | RLIMIT_CPU (all POSIX tiers ENFORCED) + wall-timeout + process-group / uid kill (all tiers ENFORCED) |
| T3 | Exceed **wall time** (sleep, deadlock, network stall) | `time.sleep(1e9)`; blocking socket | parent wall-clock `communicate(timeout=)` -> SIGKILL of the whole group (all tiers ENFORCED). This is the universal backstop that catches anything the rlimits miss. |
| T4 | **Read files** (keys, other runs' scratch, the sealed test if on disk) | `open("/root/.attestera/key")`; `np.load(...)`; numpy C-ext I/O | scrubbed env (no secrets in child env, all tiers) + tempdir-confined cwd (all tiers) + dedicated low-priv uid so host files are unreadable by perms (T2/T3 ENFORCED) + read-only root FS (T3 ENFORCED) + AST deny of `open`/numpy-I/O attrs (all tiers ADVISORY). Secrets must never be readable-by-uid (T2/T3). |
| T5 | **Write files** outside scratch (beacons, poison) | write to `/tmp`, `/dev/shm`, the network volume | RLIMIT_FSIZE (T2/T3 ENFORCED) + tempdir cwd + uid-owned-only scratch + out-of-scratch uid-file purge on teardown (T2/T3 ENFORCED) + read-only root FS except one scratch mount (T3 ENFORCED) |
| T6 | **Exfiltrate via network** | `urllib`/`socket` POST of the sealed features/keys | no-network namespace / `--network=none` (T3 ENFORCED) + AST deny of socket/urllib/requests (all tiers ADVISORY) + nothing-sensitive-to-exfil argument (T1/T2: scrubbed env + secrets unreadable + only non-secret train data crosses). **Network egress is NOT blocked at T1/T2 on a stock host.** Stated plainly. |
| T7 | **Fork-bomb** / spawn detached daemons | `os.fork()` loop; `setsid` away from the group | RLIMIT_NPROC (T2/T3 ENFORCED) + process-group kill (all POSIX ENFORCED for in-group children) + **kill-by-uid reap loop** for processes that `setsid` out of the group (T2/T3 ENFORCED) + pids cgroup limit (T3 ENFORCED) |
| T8 | **Import dangerous modules** | `import os, socket, ctypes, subprocess` | AST import allow-list applied before exec (all tiers ADVISORY) + restricted `__import__` in the child namespace (all tiers, ADVISORY: bypassable via C-ext) + the OS-level controls above which make a successful import *harmless* (T2/T3) |
| T9 | **Leak the sealed test** during selection | read the sealed rows from memory/disk; index-based label leakage | structural: the sealed split is never serialized into the job the child loads (section 6) ENFORCED; spawn-not-fork so the child inherits no parent memory ENFORCED at T2/T3; behavioral permutation-equivariance probe ADVISORY |
| T10 | **Escape the interpreter** via a CPython / numpy C-extension exploit | a memory-corruption bug reachable through allowed numpy/sklearn calls | seccomp syscall filter (T3, partial: shrinks kernel attack surface) + no-new-privs (T2/T3 ENFORCED, kills SUID escalation) + dedicated uid (blast radius is one throwaway uid) + ultimately: run on a host you can throw away (T3). **No tier fully closes a novel C-ext interpreter exploit; stated plainly in section 7.** |
| T11 | **Return a poisoned result** to confuse the parent (NaN/Inf, symlink, FIFO, hardlink, decompression bomb) | child writes a symlink/FIFO at the predictions path; emits Inf | confused-deputy-resistant result read in the parent: `O_NOFOLLOW` + `O_NONBLOCK` + `S_ISREG` + `st_uid==run-uid` + `st_nlink==1` + size cap + finite-vector validation (T2/T3 ENFORCED; T1 gets the size cap + finite check) |

Mapping principle: **AST checks (T8, parts of T4/T6) are ADVISORY, not a security boundary.**
They reduce accidental and unsophisticated-adversarial damage and they keep the audit log
readable, but a determined adversary reaches file/network through numpy/sklearn C extensions
that need none of the banned Python names (this is a *confirmed* finding in
`vfplatform/authoring.py`: numpy's own `np.load`/`np.savetxt`/`memmap` were used to read and
exfiltrate a key with no banned import). The OS-level controls (T2/T3) are the actual boundary.

---

## 2. Tier 1 (local, now): hardened subprocess + advisory AST gate

Tier 1 runs on the developer/CI host (often macOS). It extends the existing
`frontier/sandbox.py` without changing its public signature. Everything it adds is either
ENFORCED by the OS or honestly labeled BEST-EFFORT / ADVISORY.

### 2.1 What Tier 1 enforces

ENFORCED (POSIX):
- `RLIMIT_CPU = (cpu_seconds, cpu_seconds+1)` in `preexec_fn` (already present). Defeats T2.
- Wall-clock timeout via `proc.communicate(timeout=wall_seconds)` -> `os.killpg(pgid, SIGKILL)`.
  Universal backstop for T1/T2/T3 (already present). Defeats the spin/sleep/deadlock class.
- `start_new_session=True` -> own session/process group; in-group children die with the kill
  (already present). Partial T7.

ADDED in Tier 1, ENFORCED:
- **Tempdir-confined working dir.** The child is launched with `cwd=<the TemporaryDirectory>`
  and `env` carrying `HOME`, `TMPDIR`, `TEMP`, `TMP` all set to that dir, so any relative path,
  `tempfile.*` call, or matplotlib/cache write lands inside the auto-deleted scratch dir, not
  the user's home or `/tmp`. Reduces T5 surface. (The dir is already created; this wires `cwd`
  and `env` to it.)
- **Scrubbed child environment.** The child env is rebuilt from a minimal allow-list
  (`PATH=/usr/bin:/bin`, the four TMP vars, `HOME`, and the BLAS/OMP thread caps below); every
  variable whose name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `ANTHROPIC`, `OPENAI`,
  `AWS`, `GH` is dropped. Defeats env-based key exfiltration (part of T4/T6). ENFORCED (it is a
  parent-side dict; the child cannot un-scrub it).
- **BLAS/OMP thread + arena caps.** `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=
  NUMEXPR_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`. Not a security control but mandatory for
  predictable RLIMIT_CPU accounting on many-core hosts (the pod sandbox documents why: numpy
  otherwise spawns one thread + malloc arena per core at import and burns the CPU budget before
  the candidate runs). ENFORCED (env).
- **Predictions size cap + finite validation in the parent.** The parent caps the `.npy` it
  loads (e.g. 64 MB, matching the pod's `_RESULT_MAX_BYTES`) and validates that classification
  predictions are in `range(n_classes)` and regression predictions are finite floats before
  handing them to the scorer. Defuses the decompression/quadratic-parse-bomb and NaN classes of
  T11. ENFORCED.

ADDED in Tier 1, BEST-EFFORT (honestly labeled):
- `RLIMIT_AS = address_mb` on Linux only. **On macOS (`sys.platform == "darwin"`) RLIMIT_AS is
  not honored by the kernel and can break interpreter startup, so it is skipped** (already the
  case in `_preexec`). On macOS the *only* enforced memory control is the wall-clock timeout
  (a memory bomb degrades to a timeout). This is logged as `mem_guard="wall-timeout-only"` so we
  never claim a memory cap we did not set. Defeats T1 only on Linux.

ADDED in Tier 1, ADVISORY (explicitly NOT a security boundary):
- **AST allow-list / import policy applied before execution.** Before the candidate string is
  written to disk for the runner, the parent parses it with `ast.parse` and rejects on policy
  violation, returning `RunResult(ok=False, error_kind="policy", error=<violation>)`. This is
  the same machinery as `vfplatform/authoring.py:estimator_static_check`, reused conceptually
  (not imported, to keep `frontier/` self-contained per ROADMAP). Its purpose is early, cheap,
  auditable rejection of obviously-hostile or obviously-broken code and a clean error taxonomy,
  **not** containment. The OS controls contain; the AST gate triages.

  Allow / deny lists (Tier-1 estimator policy):

  - **Import roots ALLOWED:** `numpy`, `np`, `scipy`, `sklearn`, `math`, `statistics`,
    `itertools`, `functools`, `operator`, `collections`, `numbers`, `warnings`, `typing`,
    `dataclasses`, `random`. (Torch/jax added per Tier-3 GPU policy in section 4; not allowed at
    Tier 1 unless the host opts in.)
  - **sklearn submodules DENIED:** `sklearn.datasets`, `sklearn.model_selection`,
    `sklearn.externals`. A model has no business loading external data or peeking at a split.
  - **Import roots DENIED (everything not on the allow-list), with these called out explicitly
    for a specific error message:** `os`, `sys`, `subprocess`, `socket`, `ssl`, `builtins`,
    `importlib`, `ctypes`, `cffi`, `pickle`, `marshal`, `shelve`, `shutil`, `pathlib`,
    `requests`, `urllib`, `http`, `ftplib`, `smtplib`, `io`, `tempfile`, `threading`,
    `asyncio`, `signal`, `resource`, `multiprocessing`, `mmap`, `fcntl`, `pty`, `code`, `pdb`,
    `inspect`, `gc`, `atexit`, `webbrowser`, `platform`, `getpass`.
  - **Banned names (Name nodes):** `eval`, `exec`, `compile`, `open`, `__import__`, `globals`,
    `locals`, `vars`, `getattr`, `setattr`, `delattr`, `__builtins__`, `breakpoint`, `input`,
    `memoryview`.
  - **Banned attributes (Attribute nodes):** any dunder attribute (name matches `__.*__`),
    plus the numpy/C-extension file-I/O and frame-introspection attrs that need no banned import
    (the confirmed `authoring.py` vector): `save`, `savez`, `savez_compressed`, `savetxt`,
    `load`, `loadtxt`, `genfromtxt`, `fromfile`, `tofile`, `memmap`, `ctypeslib`, `fromregex`,
    `DataSource`, `frombuffer`, `getbuffer`, `fromiter`, `f_back`, `f_globals`, `f_locals`,
    `f_builtins`, `f_code`, `gi_frame`, `gi_code`, `cr_frame`.
  - **Structural rejects:** relative imports (`level != 0`); `global`/`nonlocal`; constant-true
    `while` with no `break` (infinite-loop guard); AST node count > 4000; code length > a cap
    (e.g. 32k chars for neural-net code). Require a callable `build_estimator`.

  The runner child *also* execs under restricted `__builtins__` and a restricted `__import__`
  (as in `authoring.py`) so an allowed-but-misused dynamic import is blocked at runtime too.
  Both layers are ADVISORY; their failure mode (a C-ext exploit) is covered only by T2/T3.

### 2.2 Tier-1 firewall (unchanged, restated)

The runner returns `OK` + a predictions `.npy`, or `ERR:<kind>:<msg>`. It never returns a
metric. `error_kind` taxonomy extends the existing
`timeout|import|fit|build|oom|cpu|other` with `policy` (AST rejection) and `result`
(failed result validation). The parent computes every score. Firewall preserved.

### 2.3 Tier-1 honest ceiling

Tier 1 gives **resource + crash isolation and an advisory policy gate, but NOT network
isolation and NOT memory enforcement on macOS, and NOT containment against a C-extension
exploit.** It is appropriate for trusted-ish code on a dev/CI host. It is NOT appropriate for
running maximally-untrusted code on a host that holds secrets or production data. That is what
Tier 2/3 are for. The active-tier log records `tier="local"`, `network="not-isolated"`,
`mem_guard=<"rlimit-as"|"wall-timeout-only">`.

---

## 3. Tier 2 (Linux host): dedicated uid + no-new-privs + seccomp + no-network + RO FS

Tier 2 is the `vfplatform/authored_pod_sandbox.py` pattern, generalized into the `frontier`
runner contract. It runs on a Linux host the project controls (a RunPod/Prime Intellect pod
without Docker, or a bare VM). Everything here is ENFORCED by the Linux kernel except where
noted. The parent process is trusted (root or a sudo-capable launcher).

### 3.1 Per-run dedicated low-privilege uid (ENFORCED)

- A pool of dedicated uids (`sbx1..sbxN`, gid `sbx`) is provisioned once on the host. Each run
  `flock`s a free slot so the uid is *exclusively* held for the run's lifetime. This (a) lets
  us reap every process of that uid even if it `setsid`s out of the process group (defeats the
  detached-daemon half of T7), and (b) isolates concurrent runs from each other (a shared
  `nobody` would let one run `ptrace`/kill/read another's scratch).
- The child is launched via:
  `setpriv --reuid sbxK --regid sbx --clear-groups --no-new-privs --inh-caps=-all --ambient-caps=-all <python> runner.py ...`
  - `--reuid/--regid/--clear-groups`: drop to the unprivileged uid; host files (keys, other
    scratch, the sealed test if ever on disk) are unreadable by Unix perms. Defeats T4.
  - `--no-new-privs`: even if the child execs a SUID-root or PATH-hijacked binary, the kernel
    grants no new privileges. Kills the SUID/PATH-hijack escalation class (part of T10).
  - `--inh-caps=-all --ambient-caps=-all`: no capability survives the uid drop.
- Secrets live root-600 in a directory the sandbox uid cannot read (`/root/.attestera` in the
  pod). They are never on a path the child can open. The child env is scrubbed (no
  `ANTHROPIC_API_KEY` etc.).

### 3.2 rlimits (ENFORCED on Linux)

Set in the child driver before exec (the pod sets them in `_DRIVER`):
- `RLIMIT_CPU` (T2), `RLIMIT_AS` + `RLIMIT_DATA` (T1, *enforced here* unlike macOS),
  `RLIMIT_FSIZE` (T5), `RLIMIT_NPROC` (T7 fork-bomb guard).

### 3.3 No-network (ENFORCED where the substrate allows)

- Preferred: launch the child in a network namespace with no interfaces
  (`unshare --net`), or apply `iptables`/`nft` owner-match egress drop for the sandbox uid.
- If the host provides neither (the documented pod case: no namespaces/iptables-owner), Tier 2
  **degrades to "no-network NOT enforced"** and relies on the nothing-sensitive-to-exfil
  argument (scrubbed env + secrets unreadable + only non-secret train data crosses the
  boundary). This degradation is logged as `network="not-isolated:no-netns"` and the seccomp
  socket-block below is the partial mitigation. We never claim network isolation we did not get.
- When `libseccomp` is available (see 3.4) the seccomp filter denies `socket`/`connect`,
  which closes most egress even without a netns. Labeled accordingly.

### 3.4 seccomp / syscall filter sketch (ENFORCED where libseccomp present; partial T6/T10)

Apply a seccomp-bpf filter in `preexec_fn` (or via a tiny `seccomp` shim before `exec`) using a
**default-deny-then-allow** policy. Sketch (pyseccomp / libseccomp):

```
filter = SyscallFilter(defaction=KILL_PROCESS)   # default: kill on any unlisted syscall
# compute / memory / fs-read needed by numpy+sklearn (+torch at T3):
for s in ("read","write","readv","writev","close","fstat","lseek","mmap","mprotect","munmap",
          "brk","rt_sigaction","rt_sigprocmask","rt_sigreturn","ioctl","openat","newfstatat",
          "getdents64","getrandom","futex","clock_gettime","gettimeofday","sched_yield",
          "exit","exit_group","madvise","set_robust_list","prlimit64","sysinfo","uname",
          "nanosleep","clock_nanosleep","getpid","gettid"):
    filter.add_rule(ALLOW, s)
# explicitly DENY the egress + escalation classes (redundant with default-deny but auditable):
for s in ("socket","connect","bind","sendto","sendmsg","accept","accept4",
          "ptrace","process_vm_readv","process_vm_writev","kexec_load",
          "execve","execveat","fork","vfork","clone","clone3","unshare","setns"):
    filter.add_rule(KILL_PROCESS, s)
filter.load()
```

Notes and honesty:
- `openat` is allowed (numpy/sklearn open shared libs and data files); FS *confinement* is
  therefore done by the uid + RO-FS + scratch-mount layers, **not** by seccomp. seccomp's job
  here is to kill egress (`socket`/`connect`) and the escalation/forking classes.
- `clone`/`fork` are denied -> single-process candidate. If a candidate legitimately needs
  threads (BLAS), allow `clone` with a `CLONE_THREAD`-only argument filter, or keep threads
  capped to 1 via the BLAS env (preferred; simpler; matches the pod). A torch DataLoader with
  workers needs `clone`; the Tier-3 GPU policy relaxes this with an argument filter, documented.
- seccomp **shrinks** the kernel attack surface for T10; it does not *eliminate* a C-ext exploit
  reachable through the allowed syscalls. Stated plainly in section 7.

### 3.5 Read-only FS except a scratch dir (ENFORCED at T2 via perms; fully at T3)

At Tier 2 without mount namespaces, "read-only root FS" is approximated by: the sandbox uid
owns nothing outside its `0700` scratch dir, system dirs are not uid-writable, and
`RLIMIT_FSIZE` + the post-run uid-file purge (`find /tmp /var/tmp /dev/shm -uid <uid> -delete`)
clean up any beacon the uid wrote outside scratch. A true read-only bind-mounted rootfs is a
Tier-3 (container) property; Tier 2 logs `fs="uid-confined"` not `fs="readonly-root"`.

### 3.6 Teardown (ENFORCED)

On every exit path (success, error, timeout): kill-by-uid reap loop until no process of the run
uid remains (defeats fork-races that spawn a child in the enumerate->signal window), purge
out-of-scratch uid files, `rmtree` the scratch dir, release the slot lock. This is exactly the
`authored_pod_sandbox._reap_uid` + `_purge_uid_files` + `finally` pattern.

### 3.7 Result read (ENFORCED)

The parent (root) reads the child's predictions with the confused-deputy guards from
`authored_pod_sandbox._read_result`: `O_NOFOLLOW` (reject child-planted symlink), `O_NONBLOCK`
(never hang on a FIFO/device), `S_ISREG`, `st_uid == run-uid` (reject hardlink to a root file),
`st_nlink == 1`, size cap, JSON `parse_constant` that rejects `Infinity`/`NaN`, finite-vector
validation. Defeats T11.

Tier-2 active-tier log: `tier="linux-uid"`, plus the actual sub-states
(`network=<enforced|not-isolated:no-netns|seccomp-socket-block>`, `seccomp=<on|unavailable>`,
`fs="uid-confined"`).

---

## 4. Tier 3 (containerized / Prime Intellect pod): the GPU path

Tier 3 runs the *same* `runner.py` inside a no-network, resource-capped container/pod. This is
the maximally-untrusted tier and the only one that can safely run code on a host that also holds
secrets or production data, and the only GPU tier.

### 4.1 Container/pod configuration (ENFORCED by the container runtime / orchestrator)

- `--network none` (no NIC in the container; the strongest egress block). T6 ENFORCED.
- Read-only root FS (`--read-only`) plus a single writable scratch mount
  (`--tmpfs /scratch:size=<cap>,mode=0700` or a bind-mounted ephemeral volume). T5 ENFORCED;
  this is the real "read-only FS except scratch" promised at Tier 2 only by approximation.
- Memory cgroup (`--memory`, `--memory-swap` equal to disable swap), CPU quota (`--cpus`),
  pids limit (`--pids-limit`). T1/T2/T7 ENFORCED at the cgroup level, independent of in-process
  rlimits (defense in depth: a candidate that defeats RLIMIT_AS still hits the cgroup).
- `--cap-drop ALL`, `--security-opt no-new-privileges`, run as a non-root uid inside the
  container, `--read-only` plus `--security-opt seccomp=<profile.json>` (the section-3.4 filter,
  relaxed for the GPU device syscalls). T10 surface-reduction ENFORCED.
- Inside the container the child *still* runs the section-3 uid-drop + rlimits + result-read
  guards (belt and suspenders: the container is the outer jail, the uid drop is the inner jail).
- Secrets are simply not mounted into the container. The image is built without them; the LLM
  API key lives only in the trusted parent that talks to the orchestrator, never in the pod that
  runs candidate code.

### 4.2 GPU path

- The pod is requested with a GPU (Prime Intellect / RunPod GPU lane). The NVIDIA device nodes
  (`/dev/nvidia*`) and the CUDA libraries are the *only* additional host resources exposed; they
  are read-mounted, not the host FS.
- The seccomp profile is relaxed to allow the `ioctl` / `mmap` patterns the CUDA driver needs
  and to allow `clone` (DataLoader workers, NCCL) under an argument filter; this widening is
  documented in the profile and logged (`seccomp="gpu-profile"`), because a wider syscall set is
  a weaker T10 guarantee and we will not log it as the tabular profile.
- GPU memory is capped per-process where the framework supports it
  (`torch.cuda.set_per_process_memory_fraction`, `XLA_PYTHON_CLIENT_MEM_FRACTION` for jax), and
  the pod itself is a fixed-size GPU allocation. A GPU OOM degrades to a caught
  `RunResult(error_kind="oom")`, never a host crash.
- The `$5/day` spend cap (from the deployment notes) is enforced *outside* the sandbox by the
  orchestrator's budget guard: the sandbox cannot request a pod the budget guard has not
  approved. This keeps a runaway proposal loop from being a financial DoS.

### 4.3 Substrate swap, identical contract

The container/pod runs the same `runner.py` against the same `job.npz` (train + eval features)
and writes the same predictions artifact. The parent calls the *same* `run_program(...)`; only
the launcher underneath differs (subprocess vs `setpriv` vs `docker run`/pod-create). I3 holds.
Tier-3 active-tier log: `tier="container"` or `tier="pod-gpu"`, `network="none"`,
`fs="readonly-root+scratch"`, `seccomp=<"profile"|"gpu-profile">`, `mem_guard="cgroup+rlimit"`.

---

## 5. `SandboxPolicy`: tier selection + honest degradation

A single abstraction lets the engine pick a tier by trust and availability and *prove* in the
log which protections actually held. The runner contract (`run_program`) is unchanged; the
policy decides *which substrate* `run_program` dispatches to and records what it enforced.

```python
# frontier/core/sandbox_policy.py  (design sketch; not yet implemented)
from dataclasses import dataclass, field
from enum import Enum

class Tier(str, Enum):
    LOCAL = "local"            # Tier 1: subprocess + rlimits + advisory AST gate
    LINUX_UID = "linux-uid"    # Tier 2: + setpriv uid + seccomp + (maybe) netns + uid-confined fs
    CONTAINER = "container"    # Tier 3: no-network container, RO-root + scratch, cgroups
    POD_GPU = "pod-gpu"        # Tier 3: container on a GPU pod (relaxed seccomp profile)

@dataclass(frozen=True)
class EnforcedGuarantees:
    """What the ACTIVE substrate actually enforced. Every field is set from a runtime probe,
    never from the requested tier. This dict is attached to every RunResult and logged."""
    tier: Tier
    cpu_rlimit: bool            # RLIMIT_CPU set
    mem_guard: str              # "rlimit-as" | "cgroup+rlimit" | "wall-timeout-only"
    wall_timeout: bool          # parent wall clock active (always True)
    network: str                # "none" | "netns" | "seccomp-socket-block" | "not-isolated"
    fs: str                     # "readonly-root+scratch" | "uid-confined" | "tempdir-cwd"
    uid_drop: bool              # setpriv to a dedicated low-priv uid
    no_new_privs: bool
    seccomp: str                # "profile" | "gpu-profile" | "unavailable"
    fork_guard: str             # "nproc+pidcg+uid-reap" | "nproc+pgkill" | "pgkill"
    advisory_ast_gate: bool     # AST policy ran (ADVISORY, not a boundary)

class SandboxPolicy:
    def __init__(self, requested: Tier, *, untrusted: bool = True):
        self.requested = requested
        self.untrusted = untrusted

    def resolve(self) -> Tier:
        """Pick the strongest tier that is AVAILABLE on this host, capped by what was requested.
        Probe-driven: container present? setpriv + dedicated uids present? GPU present?
        DEGRADES DOWNWARD honestly and LOGS a WARNING when the requested tier is unavailable.
        Crucially: if untrusted=True and the resolved tier is LOCAL, it logs a LOUD warning that
        maximally-untrusted code is about to run with resource-only isolation, and (configurable)
        may REFUSE rather than run."""
        ...

    def run(self, program, X_train, y_train, X_eval, *, kind, **rl) -> "RunResult":
        tier = self.resolve()
        result = _DISPATCH[tier](program, X_train, y_train, X_eval, kind=kind, **rl)
        result.enforced = self._probe_enforced(tier)   # set from what actually happened
        _audit_log(result.enforced)                     # one structured line per run
        return result
```

Honesty rules baked into the abstraction:
1. `EnforcedGuarantees` is populated from **runtime probes of what the substrate did**, not from
   the requested tier. If `unshare --net` failed, `network="not-isolated"` even though Tier 2
   was requested. The log reflects reality.
2. `resolve()` only ever degrades *downward* and logs the degradation. It never silently
   upgrades a claim. A run that requested CONTAINER but landed on LOCAL is logged as LOCAL.
3. A policy of `untrusted=True` resolving to `LOCAL` emits a prominent warning, and in
   "strict" mode raises rather than runs (so CI for the CORE can require T2+ for untrusted code).
4. The certificate / result writer reads `RunResult.enforced` and stamps the certificate with
   the *enforced* tier, so a downstream reader can see "this number was produced under
   resource-only isolation" and weight it accordingly. The sandbox tier becomes part of the
   provenance, exactly like the code digest already is.

This keeps I4 (honest tiering) mechanical rather than aspirational.

---

## 6. Sealed-test protection (the load-bearing guarantee)

The sealed test is the integrity anchor of the whole system (ROADMAP invariant: sealed touched
once, for the winner). The sandbox's job is to make it **structurally impossible** for authored
code to see sealed labels (and, during selection, sealed contents at all). Layers:

S1. **The job that crosses the boundary never contains the sealed split's labels, and never
    contains the sealed split's features during selection.** `run_program` is called with
    `(X_train, y_train, X_eval)` only. `X_eval` is the *features* of the split the candidate
    must predict (the validation features during selection; the sealed features only for the
    single winner-certification call). `y_eval` (the labels that score the predictions) is
    **never** serialized into `job.npz`. The parent holds the labels and scores the returned
    predictions. ENFORCED by the runner contract itself; it is not a check, it is the shape of
    the data that crosses. (`frontier/sandbox.py` already only packs `Xtr, ytr, Xev`.)

S2. **During selection the candidate never even receives sealed features.** Selection scores on
    validation. The sealed features are passed to `run_program` exactly once, for the chosen
    winner, in the single certify call. So even the sealed *inputs* are invisible to every
    candidate during the search; a candidate cannot fit-to or memorize the sealed inputs because
    it never sees them. This is why "select-then-certify" keeps the certificate at one counted
    peek no matter how many candidates were searched.

S3. **Fresh interpreter (spawn / new process), not fork.** The child is a brand-new process
    image (`subprocess` at T1; spawn/new container at T2/T3), so it inherits **none** of the
    parent's in-memory objects, including the in-memory sealed `SealedTest` and its rows. There
    is no Python reference reachable from the child to the sealed data. ENFORCED.

S4. **Sealed data is not on a path the child can read.** At T2/T3 the sealed split, if ever
    materialized to disk, lives outside the uid-readable scratch (root-600 / not mounted). The
    child's only readable inputs are the `job.npz` (train + the eval features it must predict)
    and its own scratch. ENFORCED at T2/T3.

S5. **One-peek guard stays in the trusted parent.** The frozen `vfplatform.sealed.SealedTest`
    (`max_peeks=1`, raises on a second peek) and `certify_on_sealed` run in the *parent*, never
    in the sandbox. The sandbox only ever produces predictions; the parent decides whether to
    spend the single peek. The sandbox cannot consume, increment, or bypass the peek counter
    because it has no handle to it (consequence of S3).

S6. **Behavioral leakage probe (ADVISORY, defense in depth).** Before trusting a candidate's
    predictions, the parent may run the permutation-equivariance probe from `authoring.py`
    (`predict(X[perm]) == predict(X)[perm]`): a predictor that leaked labels by row index rather
    than computing from features fails it. This is ADVISORY (it catches a specific index-based
    leak, not all leakage) and is labeled as such; the structural guarantees S1-S5 are the
    actual protection.

Net: authored code cannot see sealed labels at any tier (S1, S5 are structural), cannot see
sealed features during selection at any tier (S2 is structural), and at T2/T3 cannot reach the
sealed data on disk or in memory even adversarially (S3, S4). The residual at T1 is that a
C-ext exploit could in principle read parent memory, but at T1 the sealed rows are not even in
the runner's process (separate subprocess), so the exploit would have to cross a process
boundary, which is the T10 escape problem, not a sealed-specific hole.

---

## 7. Innovations + residual risks per tier (stated plainly)

### Innovations

N1. **Enforced-not-requested provenance (`EnforcedGuarantees` stamped on the certificate).**
    The certificate records the isolation that *actually held at runtime*, probed, not the tier
    that was asked for. This makes "we ran untrusted code safely" a falsifiable, auditable claim
    rather than a marketing line, and lets a reviewer discount a number produced under weaker
    isolation. I have not seen an AutoML/autoresearch sandbox that carries its own enforcement
    evidence into the result artifact; here it is mechanical (section 5, rule 4).

N2. **The firewall is the data shape, not a check.** Because the sealed labels are simply never
    serialized into the boundary payload (S1) and the metric is only ever computed in the
    parent, the predictions-only firewall and the sealed-blindness are *structural properties of
    what crosses the boundary*, not runtime guards that could be misconfigured or bypassed.
    There is nothing to turn off. This is strictly stronger than an in-process exec with a
    "please don't peek" convention (the bug the spine replaced).

N3. **Two-jail nesting at T3 with one runner.** The same `runner.py` runs the section-3 uid +
    rlimit + confused-deputy result read *inside* the section-4 container (outer cgroup + netns +
    RO-FS + seccomp jail). A failure of either layer alone does not breach: a candidate that
    defeats RLIMIT_AS still hits the memory cgroup; one that escapes the uid still has no NIC and
    no host FS. The nesting costs nothing because the inner guards are already in the runner that
    every tier shares (I3).

### Residual risks each tier does NOT cover (no overclaiming)

Tier 1 (local) does NOT cover:
- R1. **Network egress.** A stock dev/CI host has a live NIC; the AST gate is advisory and a
  numpy/C-ext path can open a socket. T6 is *not* enforced at T1. Mitigation is the
  nothing-sensitive-to-exfil argument (scrubbed env) plus moving untrusted code to T2/T3.
- R2. **Memory enforcement on macOS.** RLIMIT_AS is unenforced on Darwin; a memory bomb is only
  caught by the wall-clock timeout (degrades to a timeout, may swap the host first). Logged as
  `mem_guard="wall-timeout-only"`.
- R3. **C-extension interpreter escape.** A memory-corruption exploit reachable through allowed
  numpy/sklearn C code is not contained by an AST gate or rlimits. At T1 the blast radius is the
  dev host's user account.

Tier 2 (Linux uid) does NOT cover:
- R4. **Network egress when the host lacks netns/iptables-owner** (the documented pod case).
  Degrades to `network="not-isolated:no-netns"`; the seccomp `socket`/`connect` deny is the
  partial mitigation, but if `libseccomp` is also absent, egress is open and we say so.
- R5. **A C-ext exploit that the seccomp allow-list permits.** seccomp shrinks but does not
  eliminate the kernel attack surface; `openat`/`mmap`/`ioctl` are allowed for numpy to function,
  so a kernel bug reachable through them is reachable. Blast radius is one throwaway uid on a pod,
  not a secret-bearing host (secrets are root-600 and unreadable by the uid).
- R6. **Kernel-level privilege escalation** (a local-root CVE in the host kernel). uid-drop +
  no-new-privs + cap-drop raise the bar but cannot patch the kernel. Mitigation is a throwaway
  pod and a patched host.

Tier 3 (container / GPU pod) does NOT cover:
- R7. **Container escape via a runtime / kernel CVE** (a runc/CUDA-driver/kernel bug). `--network
  none` + RO-FS + seccomp + cap-drop make this hard and make a successful escape land on a host
  with no secrets mounted and no network, but a kernel/runtime 0-day is out of scope of any
  in-band control. Mitigation: ephemeral pods, patched runtime, no secrets co-resident.
- R8. **Wider GPU seccomp profile.** The GPU profile must allow more syscalls (CUDA ioctls,
  `clone` for workers) than the tabular profile, so the T10 guarantee is *weaker* on the GPU
  lane. This is logged as `seccomp="gpu-profile"` so it is never conflated with the tighter CPU
  profile.
- R9. **Covert/timing channels and resource-side-channels** between co-tenant pods are not
  addressed by any tier; they are out of scope for this system (the threat is exfiltration of the
  sealed test, which is structurally absent from the boundary, S1/S2, so a timing channel has
  nothing high-value to leak during selection).

Cross-tier residual, stated once: **an AST allow-list is a triage and audit tool, never a
security boundary** (confirmed numpy-I/O bypass in `authoring.py`). Every place this document
relies on containment, it relies on the OS-level controls of T2/T3, and it says so.

---

## 8. Implementation order (maps to ROADMAP phases)

1. **Now (extends Phase 0):** Tier-1 additions to `frontier/sandbox.py` that need no new
   substrate: tempdir `cwd` + scrubbed `env` + BLAS caps; predictions size cap + finite
   validation in the parent; the advisory AST gate as a pre-exec policy returning
   `error_kind="policy"`; restricted `__builtins__`/`__import__` in the runner child. Add
   `RunResult.enforced` and the structured audit log. (No public-signature change; I3 holds.)
2. **Tier 2 module** (`frontier/core/sandbox_linux.py`): the `setpriv` uid-pool +
   `--no-new-privs` + rlimits (incl. NPROC) + uid-reap teardown + confused-deputy result read +
   seccomp filter + best-effort netns, generalizing `authored_pod_sandbox.py` into the
   `run_program` contract. Probe-driven `EnforcedGuarantees`.
3. **Tier 3 module** (`frontier/core/sandbox_container.py`): `docker run`/pod-create launcher
   (no-network, RO-root + scratch tmpfs, cgroup caps, seccomp profile) running the same
   `runner.py`; GPU variant with the relaxed profile and per-process GPU memory cap; wired to the
   orchestrator budget guard.
4. **`SandboxPolicy`** (`frontier/core/sandbox_policy.py`): tier resolution by probe, downward-only
   degradation, strict-mode refusal for untrusted-on-LOCAL, certificate stamping.

Each module ships with a self-test that *attacks* it (fork bomb, mem bomb, infinite loop, a
candidate that tries `open`/`socket`/`np.load` of a planted secret, a symlink/FIFO at the result
path) and asserts the run is contained and the `enforced` log matches reality. A residual risk
listed in section 7 is not a bug to hide; it is a row in the self-test marked "expected-uncovered
at this tier."
