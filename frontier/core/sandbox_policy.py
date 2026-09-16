"""frontier.core.sandbox_policy -- tiered sandbox abstraction with HONEST, PROBE-DERIVED
enforcement provenance (design 05).

This module is the policy layer the integrator puts in front of the Phase-0 runner. It does
NOT replace `frontier/sandbox.py`; it WRAPS it. The frozen Phase-0 `run_program(...)` is the
Tier-1 substrate (subprocess + RLIMIT_CPU + best-effort RLIMIT_AS + wall-kill + predictions-only
firewall). This file adds, on top of that already-working substrate:

  * `Tier` / `EnforcedGuarantees`           -- the typed provenance the certificate stamps.
  * `probe_host()`                           -- one-shot capability probe of THIS host/OS.
  * `SandboxPolicy`                          -- requested-tier -> resolved-tier (downward-only,
                                                announced) -> dispatch -> stamp what ACTUALLY held.
  * `AstPolicyChecker` / `ast_check(...)`    -- the ADVISORY allow/deny AST triage (NOT a boundary).
  * `scrub_env(...)`                         -- the env scrubber usable by execution/authoring.
  * `validate_predictions(...)`              -- the parent-side finite/size/range guard (T11).

It also carries the Tier-2 (Linux uid + seccomp + netns) and Tier-3 (container/pod) HOOKS and
their honest "not enforced on this OS" status, so the engine code is written once against one
contract and the substrate is swapped underneath without any caller change (invariant I3).

# === WIRING ===
The integrator (CoreOrchestrator, design 04) composes this as the single front door to execution:

    from frontier.core.sandbox_policy import SandboxPolicy, Tier
    from frontier import sandbox as _spine_sandbox          # the FROZEN Phase-0 runner

    pol = SandboxPolicy(requested=Tier.LINUX_UID, untrusted=True, strict=False,
                        local_runner=_spine_sandbox.run_program)   # inject; never edit the spine
    res = pol.run(program, X_train, y_train, X_eval, kind=task.kind,
                  wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds, address_mb=4096)
    # res is the SAME frontier.program.RunResult the spine already returns (firewall preserved:
    # predictions only, never a metric). res.enforced is an EnforcedGuarantees describing what
    # the ACTIVE substrate enforced on THIS host -- not what was requested.

Two integration points, both additive and contract-preserving:

  (1) BEFORE dispatch the policy may run the ADVISORY `ast_check(program.code)`. On a violation
      it returns a RunResult(ok=False, error_kind="policy") WITHOUT executing -- a cheap, auditable
      early reject and a clean error taxonomy. This is triage, NOT containment (a numpy C-ext can
      reach files/sockets with none of the banned names -- confirmed in vfplatform/authoring.py).
      The OS controls (T2/T3) contain; the AST gate triages. Default is on; set ast_gate=False to
      skip (e.g. when the caller already gated upstream).

  (2) AFTER dispatch the policy attaches `res.enforced` (an EnforcedGuarantees probed from what
      the substrate actually did) and emits ONE structured audit line. The certificate writer
      reads `res.enforced` and stamps the certificate with the ENFORCED tier, so a downstream
      reader can weight a number by the isolation that actually held (innovation N1). RunResult is
      a frozen Phase-0 dataclass we cannot edit, so `enforced` is attached as a dynamic attribute
      (Python dataclasses are not slotted here); `enforced_of(res)` reads it back safely.

Sealed-test protection is STRUCTURAL and lives in the trusted parent, not here: `run_program`
is only ever called with (X_train, y_train, X_eval-features); the scoring labels and the sealed
SealedTest one-peek guard never cross the boundary (design 05 section 6; CONTRACT invariants
1,2). This module deliberately does NOT take y_eval or a SealedTest handle -- that absence is the
firewall (innovation N2). The env scrubber here is also imported by execution/authoring so the
same secret-stripping policy applies wherever untrusted code is launched.

Honesty contract (design 05 section 5, baked in mechanically):
  H1. EnforcedGuarantees is populated from `probe_host()` + the live dispatch result, NEVER from
      the requested tier. If `unshare --net` is absent, network="not-isolated" even if T2 asked.
  H2. resolve() only ever degrades DOWNWARD and logs the degradation; it never silently upgrades.
  H3. untrusted=True resolving to LOCAL emits a LOUD warning; in strict=True it RAISES rather than
      run maximally-untrusted code under resource-only isolation.
  H4. No tier claims a guarantee it did not probe. macOS RLIMIT_AS is reported as
      mem_guard="wall-timeout-only" because the Darwin kernel does not honor it.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

_LOG = logging.getLogger("frontier.core.sandbox_policy")

# RunResult is the FROZEN Phase-0 result type. We import it so the policy returns exactly the same
# object the spine returns (firewall + taxonomy preserved). We never edit it.
from ..program import Program, RunResult


# ===================================================================== Tier + EnforcedGuarantees
class Tier(str, Enum):
    """The isolation substrate. str-Enum so it serializes cleanly into a certificate / audit log.

    Ordered weakest -> strongest by `_RANK`. `resolve()` degrades along this order, DOWNWARD only.
    """
    LOCAL = "local"            # T1: subprocess + RLIMIT_CPU + best-effort RLIMIT_AS + wall-kill + AST triage
    LINUX_UID = "linux-uid"    # T2: + setpriv dedicated uid + no-new-privs + seccomp + (maybe) netns + uid-confined fs
    CONTAINER = "container"    # T3: no-network container, RO-root + scratch tmpfs, cgroups, seccomp profile
    POD_GPU = "pod-gpu"        # T3: container on a GPU pod (RELAXED seccomp profile -> weaker T10, logged as such)


# Strength order. Degradation walks down this list; we never resolve to a stronger tier than asked.
_RANK = {Tier.LOCAL: 0, Tier.LINUX_UID: 1, Tier.CONTAINER: 2, Tier.POD_GPU: 3}


@dataclass(frozen=True)
class EnforcedGuarantees:
    """What the ACTIVE substrate actually enforced on THIS host, probed -- never the requested tier.

    Every field is set from `probe_host()` (capabilities) intersected with the resolved tier and the
    live dispatch outcome. This object is attached to every RunResult and stamped onto the certificate
    so "we ran untrusted code under isolation X" is a falsifiable, auditable claim (innovation N1).

    Field vocabularies (closed sets; documented so a reader knows exactly what a value means):
      mem_guard : "cgroup+rlimit" (T3) | "rlimit-as" (Linux T1/T2, kernel-enforced) |
                  "wall-timeout-only" (macOS T1, RLIMIT_AS unenforced -> only the wall clock catches OOM)
      network   : "none" (T3 --network none) | "netns" (T2 unshare --net) |
                  "seccomp-socket-block" (T2, libseccomp denies socket/connect) | "not-isolated" (stock host NIC live)
      fs        : "readonly-root+scratch" (T3) | "uid-confined" (T2 perms) | "tempdir-cwd" (T1 scratch cwd only)
      seccomp   : "profile" (T2/T3 tabular filter) | "gpu-profile" (T3 GPU, wider syscalls -> weaker T10) | "unavailable"
      fork_guard: "nproc+pidcg+uid-reap" (T3) | "nproc+pgkill" (T2) | "pgkill" (T1 process-group kill only)
    """
    tier: Tier
    cpu_rlimit: bool            # RLIMIT_CPU set in the child (defeats the CPU-spin class)
    mem_guard: str              # see vocabulary above
    wall_timeout: bool          # parent wall-clock backstop active (always True on every tier)
    network: str                # see vocabulary above
    fs: str                     # see vocabulary above
    uid_drop: bool              # setpriv to a dedicated low-priv uid (T2/T3 only)
    no_new_privs: bool          # --no-new-privs / no-new-privileges (kills SUID/PATH-hijack escalation)
    seccomp: str                # see vocabulary above
    fork_guard: str             # see vocabulary above
    advisory_ast_gate: bool     # the AST policy actually ran before dispatch (ADVISORY, not a boundary)
    notes: tuple = ()           # honest degradation notes ("requested CONTAINER, ran LOCAL: docker absent", ...)

    def as_dict(self) -> dict:
        """Flat, JSON-safe dict for the certificate / audit log. `tier` -> its string value."""
        return {
            "tier": self.tier.value,
            "cpu_rlimit": bool(self.cpu_rlimit),
            "mem_guard": self.mem_guard,
            "wall_timeout": bool(self.wall_timeout),
            "network": self.network,
            "fs": self.fs,
            "uid_drop": bool(self.uid_drop),
            "no_new_privs": bool(self.no_new_privs),
            "seccomp": self.seccomp,
            "fork_guard": self.fork_guard,
            "advisory_ast_gate": bool(self.advisory_ast_gate),
            "notes": list(self.notes),
        }


# ===================================================================== Host capability probe
@dataclass(frozen=True)
class HostCapabilities:
    """One-shot probe of what THIS host/OS can actually enforce. Cached at policy construction.

    These are CAPABILITIES (is the tool present? does the kernel honor the rlimit?), not claims about
    a particular run. A capability being True does not mean it was used -- only that the tier that needs
    it is *available*. The per-run EnforcedGuarantees intersects capabilities with the resolved tier.
    """
    platform: str               # sys.platform ("darwin" | "linux" | ...)
    posix: bool                 # os.name == "posix"
    rlimit_cpu: bool            # RLIMIT_CPU available (POSIX)
    rlimit_as_enforced: bool    # RLIMIT_AS HONORED by the kernel (False on Darwin even though the symbol exists)
    rlimit_nproc: bool          # RLIMIT_NPROC available (fork-bomb guard; Linux)
    has_setpriv: bool           # `setpriv` on PATH (T2 uid drop)
    has_unshare: bool           # `unshare` on PATH (T2 network namespace)
    has_seccomp: bool           # python `seccomp`/`pyseccomp` importable (T2/T3 syscall filter)
    has_docker: bool            # a container runtime (`docker`/`podman`) on PATH (T3)
    has_gpu: bool               # an NVIDIA GPU visible (`nvidia-smi` on PATH AND succeeds) (T3 pod-gpu)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _seccomp_importable() -> bool:
    """True iff a libseccomp python binding is importable. We do NOT import it for real here (it is a
    Linux-only C extension); we only check availability so the probe is honest on every OS."""
    import importlib.util
    for name in ("seccomp", "pyseccomp"):
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
    return False


def _gpu_present() -> bool:
    """True iff an NVIDIA GPU is actually usable: `nvidia-smi` on PATH AND returns 0. Presence of the
    binary alone is not enough (a host can ship the CLI with no device), so we run it with a short
    timeout. Never raises; any failure -> False (honest: no GPU claimed unless one answered)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        import subprocess
        r = subprocess.run([exe, "-L"], capture_output=True, timeout=5, text=True)
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except Exception:  # noqa: BLE001 -- any failure means "no usable GPU"
        return False


def probe_host() -> HostCapabilities:
    """Probe THIS host once. Pure capability detection; sets nothing, runs no untrusted code.

    The RLIMIT_AS honesty: the symbol `resource.RLIMIT_AS` exists on macOS, but the Darwin kernel does
    NOT enforce it (and setting it can break interpreter startup -- the Phase-0 `_preexec` already skips
    it on darwin). So `rlimit_as_enforced` is False on darwin REGARDLESS of the symbol's presence. This
    is the single most important honesty bit in the probe (design 05 R2 / H4).
    """
    plat = sys.platform
    posix = os.name == "posix"
    try:
        import resource
        has_cpu = hasattr(resource, "RLIMIT_CPU")
        has_as_symbol = hasattr(resource, "RLIMIT_AS")
        has_nproc = hasattr(resource, "RLIMIT_NPROC")
    except ImportError:
        has_cpu = has_as_symbol = has_nproc = False
    # Darwin: symbol present but kernel does not honor it. Linux: honored.
    rlimit_as_enforced = bool(has_as_symbol and plat != "darwin")
    return HostCapabilities(
        platform=plat,
        posix=posix,
        rlimit_cpu=bool(has_cpu and posix),
        rlimit_as_enforced=rlimit_as_enforced,
        rlimit_nproc=bool(has_nproc and posix),
        has_setpriv=shutil.which("setpriv") is not None,
        has_unshare=shutil.which("unshare") is not None,
        has_seccomp=_seccomp_importable(),
        has_docker=(shutil.which("docker") is not None or shutil.which("podman") is not None),
        has_gpu=_gpu_present(),
    )


# ===================================================================== Advisory AST allow/deny checker
# Allow / deny lists transcribed verbatim from design 05 section 2.1 (and consistent with the confirmed
# numpy-I/O bypass finding in vfplatform/authoring.py). ADVISORY: triage + clean error taxonomy, NOT a
# security boundary. A determined adversary reaches files/sockets through numpy/sklearn C extensions that
# need none of these names; the OS controls of T2/T3 are the actual boundary. Documented, not hidden.

ALLOWED_IMPORT_ROOTS = frozenset({
    "numpy", "np", "scipy", "sklearn", "math", "statistics", "itertools", "functools",
    "operator", "collections", "numbers", "warnings", "typing", "dataclasses", "random",
})

DENIED_SKLEARN_SUBMODULES = frozenset({
    # A model has no business loading external data or peeking at a split.
    "sklearn.datasets", "sklearn.model_selection", "sklearn.externals",
})

# Called out explicitly for a specific, auditable error message. (Anything NOT in ALLOWED_IMPORT_ROOTS is
# denied regardless; these are the high-signal ones we name in the violation string.)
DENIED_IMPORT_ROOTS = frozenset({
    "os", "sys", "subprocess", "socket", "ssl", "builtins", "importlib", "ctypes", "cffi",
    "pickle", "marshal", "shelve", "shutil", "pathlib", "requests", "urllib", "http", "ftplib",
    "smtplib", "io", "tempfile", "threading", "asyncio", "signal", "resource", "multiprocessing",
    "mmap", "fcntl", "pty", "code", "pdb", "inspect", "gc", "atexit", "webbrowser", "platform",
    "getpass",
})

BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "__builtins__", "breakpoint", "input", "memoryview",
})

# numpy/C-extension file-I/O + frame/code introspection attrs that need NO banned import (the confirmed
# authoring.py exfiltration vector). Plus any dunder attribute is rejected structurally below.
#
# IMPORTANT (the file-write escape CLASS this gate triages -- design: sandbox_honesty):
#   An allowlisted library exposes BOUND METHODS that write to an arbitrary path or serialize to bytes
#   without ever naming a banned import. Confirmed on this host: `ndarray.dump('/tmp/x')` writes a pickle
#   to an arbitrary file (247 bytes observed), `ndarray.dumps()` returns the same pickle as bytes, and
#   `scipy.io.{savemat,mmwrite,hb_write,netcdf_file,wavfile}` all write files. None of these trip an import
#   rule, and the OLD denylist missed `dump`/`dumps`/the scipy.io writers entirely. We add them below.
#
#   This denylist is ADVISORY, NOT A BOUNDARY (stated again here because it is exactly the trap): a numpy
#   C-extension can reach the filesystem through code paths that bind none of these names (e.g. an attribute
#   fetched dynamically, or a write inside a C routine). The ACTUAL containment of the file-write class is:
#     (1) goal (b) -- the firewall: the child only ever receives FEATURES and returns PREDICTIONS, so a
#         file it writes can leak the FEATURES it was already given but can NEVER read the sealed LABELS
#         or forge a score (the parent computes every decision number). A successful arbitrary write is
#         therefore HARMLESS to the certificate -- this is the property goal (b) proves.
#     (2) on a containerized host, Tier-3 read-only-root + scratch tmpfs renders the write itself inert.
#   The AST attr-deny is the cheap first triage that turns the obvious `arr.dump(path)` into a clean
#   `error_kind="policy"` reject before we ever spawn a process; it is not relied on for the firewall.
BANNED_ATTRS = frozenset({
    # array -> file (numpy ndarray bound methods; need no import)
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "ctypeslib", "fromregex", "DataSource", "frombuffer", "getbuffer", "fromiter",
    # the file-write escape CLASS the old list missed: pickle-to-path / pickle-to-bytes / scipy.io writers
    "dump", "dumps", "tobytes", "tostring",                       # ndarray serialization (dump writes a path)
    "savemat", "mmwrite", "hb_write", "netcdf_file", "wavfile",   # scipy.io file writers
    # frame/code introspection (read the parent's locals/globals -> would expose sealed labels in-proc)
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "gi_frame", "gi_code", "cr_frame",
})

_MAX_AST_NODES = 4000
_MAX_CODE_CHARS = 32_000     # design 05: generous cap for neural-net code; well above tabular recipes
_REQUIRED_ENTRYPOINT = "build_estimator"


@dataclass
class AstReport:
    """Result of the advisory AST triage. `ok` True iff zero violations. `violations` is an ordered,
    de-duplicated list of human-readable strings (audit-friendly). `has_entrypoint` flags whether a
    callable `build_estimator` was defined (the runner contract requires it)."""
    ok: bool
    violations: list = field(default_factory=list)
    has_entrypoint: bool = False
    n_nodes: int = 0

    def as_dict(self) -> dict:
        return {"ok": self.ok, "violations": list(self.violations),
                "has_entrypoint": self.has_entrypoint, "n_nodes": self.n_nodes}


def _is_constant_true(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.Name) and node.id == "True":
        return True
    return False


def _has_break(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Break) for n in ast.walk(node))


class AstPolicyChecker:
    """The advisory AST allow/deny triage (design 05 section 2.1).

    ADVISORY, NOT A SECURITY BOUNDARY. Purpose: cheap, auditable early rejection of obviously-hostile or
    obviously-broken code, and a clean `error_kind="policy"` taxonomy. Containment is the job of the OS
    controls in Tier 2/3; this gate triages. A confirmed bypass exists (numpy's own C-extension file I/O
    needs none of these banned names -- see vfplatform/authoring.py), which is exactly why we label it
    advisory and never rely on it for the firewall.

    The checker is configurable so the GPU/torch policy (design 03) can extend ALLOWED_IMPORT_ROOTS with
    {"torch","jax"} when the host opts in -- without editing this default, and without ever turning the
    advisory gate into the promoter.
    """

    def __init__(self, *, allowed_roots=ALLOWED_IMPORT_ROOTS, require_entrypoint: bool = True,
                 max_nodes: int = _MAX_AST_NODES, max_chars: int = _MAX_CODE_CHARS):
        self.allowed_roots = frozenset(allowed_roots)
        self.require_entrypoint = require_entrypoint
        self.max_nodes = int(max_nodes)
        self.max_chars = int(max_chars)

    def check(self, code: str) -> AstReport:
        """Parse `code` and apply the allow/deny policy. Never executes anything. Never raises."""
        if not isinstance(code, str) or not code.strip():
            return AstReport(False, ["empty code"])
        if len(code) > self.max_chars:
            return AstReport(False, [f"code exceeds {self.max_chars} chars ({len(code)})"])
        try:
            tree = ast.parse(code)
        except SyntaxError as ex:
            return AstReport(False, [f"syntax error: {ex}"])

        nodes = list(ast.walk(tree))
        n_nodes = len(nodes)
        if n_nodes > self.max_nodes:
            return AstReport(False, [f"too large to gate ({n_nodes} AST nodes > {self.max_nodes})"],
                             n_nodes=n_nodes)

        v: list = []
        has_entry = False

        def _import_root_violation(full: str) -> Optional[str]:
            root = full.split(".")[0]
            if root not in self.allowed_roots:
                if root in DENIED_IMPORT_ROOTS:
                    return f"import of {full!r} denied (dangerous module {root!r})"
                return f"import of {full!r} not on allow-list {sorted(self.allowed_roots)}"
            if root == "sklearn" and full in DENIED_SKLEARN_SUBMODULES:
                return f"import of {full!r} explicitly denied (data/split access)"
            return None

        for node in nodes:
            if isinstance(node, ast.Import):
                for a in node.names:
                    msg = _import_root_violation(a.name)
                    if msg:
                        v.append(msg)
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0:
                    v.append("relative import not allowed (level != 0)")
                    continue
                msg = _import_root_violation(node.module or "")
                if msg:
                    v.append(msg)
            elif isinstance(node, ast.Name):
                if node.id in BANNED_NAMES:
                    v.append(f"use of banned name {node.id!r}")
                elif node.id in DENIED_IMPORT_ROOTS:
                    # even a bare reference to a dangerous module name is conspicuous
                    v.append(f"reference to denied identifier {node.id!r}")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    v.append(f"access to dunder attribute {node.attr!r}")
                elif node.attr in BANNED_ATTRS:
                    v.append(f"access to denied attribute {node.attr!r} "
                             f"(numpy/C-ext file I/O or frame introspection)")
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                v.append("global/nonlocal not allowed")
            elif isinstance(node, ast.While):
                if _is_constant_true(node.test) and not _has_break(node):
                    v.append("constant-true while loop without break (infinite-loop guard)")
            elif isinstance(node, ast.FunctionDef) and node.name == _REQUIRED_ENTRYPOINT:
                has_entry = True

        if self.require_entrypoint and not has_entry:
            v.append(f"no callable {_REQUIRED_ENTRYPOINT!r} defined (runner contract)")

        # de-duplicate while preserving order (a code body can trip the same rule many times)
        seen, ordered = set(), []
        for msg in v:
            if msg not in seen:
                seen.add(msg)
                ordered.append(msg)
        return AstReport(ok=(len(ordered) == 0), violations=ordered, has_entrypoint=has_entry,
                         n_nodes=n_nodes)


_DEFAULT_AST = AstPolicyChecker()


def ast_check(code: str) -> AstReport:
    """Module-level convenience over the default `AstPolicyChecker`. ADVISORY triage; see the class."""
    return _DEFAULT_AST.check(code)


# ===================================================================== Env scrubber (shared)
# A minimal allow-list of env vars the child keeps, plus the BLAS/OMP thread + arena caps that are
# MANDATORY for predictable RLIMIT_CPU accounting on many-core hosts (numpy otherwise spawns one thread +
# malloc arena per core at import and burns the CPU budget before the candidate runs -- documented in
# authored_pod_sandbox.py). Any inherited variable whose NAME matches a secret marker is dropped.

_ENV_ALLOW_PREFIXES = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_")
# Substrings that mark a variable as a secret -> dropped no matter what (case-insensitive on the NAME).
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "ANTHROPIC", "OPENAI",
                   "AWS", "AZURE", "GCP", "GH_", "GITHUB", "SLACK", "STRIPE", "TWILIO",
                   "CREDENTIAL", "PRIVATE", "SESSION", "COOKIE", "BEARER", "API")

_BLAS_CAPS = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MALLOC_ARENA_MAX": "2",
}


def _is_secret_name(name: str) -> bool:
    up = name.upper()
    return any(m in up for m in _SECRET_MARKERS)


def scrub_env(base_env: Optional[dict] = None, *, scratch_dir: Optional[str] = None,
              extra: Optional[dict] = None) -> dict:
    """Build a minimal, secret-free child environment from a strict allow-list.

    Reusable by the sandbox runner AND by execution/authoring (design 05: "an env scrubber usable by
    execution/authoring") so the SAME secret-stripping policy applies wherever untrusted code launches.

    Rules:
      * Start from a fresh dict (NOT a copy of the parent env -- allow-list, not deny-list, is the safe
        default: an unknown new secret-bearing var is dropped because it is not allow-listed).
      * Keep only names matching `_ENV_ALLOW_PREFIXES`, and ONLY if not flagged secret by name.
      * Force PATH to a minimal `/usr/bin:/bin` if absent (so the child can find python but not a
        user-installed hijack earlier on PATH).
      * If `scratch_dir` is given, point HOME/TMPDIR/TEMP/TMP at it so tempfile/matplotlib/cache writes
        land in the auto-deleted scratch, not the user's home or /tmp (reduces the T5 write surface).
      * Always set the BLAS/OMP thread + arena caps (mandatory for CPU accounting; also keeps a
        sandboxed child off all cores).
      * `extra` (trusted, caller-supplied) is applied last but STILL secret-scrubbed by name, so a caller
        cannot accidentally re-inject a secret.

    Returns a brand-new dict; never mutates `base_env` or `os.environ`.
    """
    src = dict(base_env) if base_env is not None else dict(os.environ)
    out: dict = {}
    for name, val in src.items():
        if _is_secret_name(name):
            continue
        if any(name == p or name.startswith(p) for p in _ENV_ALLOW_PREFIXES):
            out[name] = val
    out.setdefault("PATH", "/usr/bin:/bin")
    if scratch_dir:
        for k in ("HOME", "TMPDIR", "TEMP", "TMP"):
            out[k] = scratch_dir
    out.update(_BLAS_CAPS)
    if extra:
        for name, val in extra.items():
            if _is_secret_name(name):
                continue
            out[name] = val
    return out


# ===================================================================== Prediction validation (T11)
_PREDS_MAX = 50_000_000          # element cap; defuses a decompression / quadratic-parse bomb of preds


def validate_predictions(preds, *, kind: str, n_expected: Optional[int] = None,
                         labels: Optional[list] = None) -> tuple:
    """Parent-side guard on the predictions a child returned (design 05 T11). Returns (ok, reason).

    Never trusts the child's vector blindly:
      * length cap (`_PREDS_MAX`) -> defuse a bomb-sized prediction array;
      * length match against `n_expected` (the eval split size) when provided;
      * classification: every prediction must be a known label (string-compared against `labels` when
        given) -> rejects out-of-range / injected labels;
      * regression: every value must be a finite float (no NaN/Inf) -> rejects the poisoned-number class.

    This is ENFORCED at every tier (it runs in the trusted parent). The OS controls cannot catch a
    finite-but-poisoned NaN; this does. Never raises.
    """
    if preds is None:
        return False, "no predictions"
    try:
        seq = list(preds)
    except TypeError:
        return False, "predictions not iterable"
    n = len(seq)
    if n == 0:
        return False, "empty predictions"
    if n > _PREDS_MAX:
        return False, f"predictions too large ({n} > {_PREDS_MAX})"
    if n_expected is not None and n != n_expected:
        return False, f"prediction count {n} != expected {n_expected}"
    if kind == "classification":
        if labels:
            allowed = {str(l) for l in labels}
            for p in seq:
                if str(p) not in allowed:
                    return False, f"prediction {p!r} not a known label"
        return True, "ok"
    # regression: finite floats only
    for p in seq:
        try:
            f = float(p)
        except (TypeError, ValueError):
            return False, f"non-numeric regression prediction {p!r}"
        if not math.isfinite(f):
            return False, f"non-finite regression prediction {p!r}"
    return True, "ok"


# ===================================================================== Tier 2/3 hooks (honest status)
# These are the HOOKS + honest status the design requires. On a host that lacks the substrate (every
# field probed), they report "not enforced on this OS" and the policy degrades downward. The real Tier-2
# (setpriv/seccomp/netns/uid-reap, generalizing authored_pod_sandbox.py) and Tier-3 (docker/pod launcher)
# substrates are separate modules (design 05 section 8 steps 2,3: sandbox_linux.py, sandbox_container.py)
# that register here via `register_substrate`. Until they register, requesting T2/T3 degrades to T1 with
# a logged, stamped note -- NEVER a silent claim that T2/T3 ran.

# A substrate is a callable with the SAME signature as the Phase-0 run_program (invariant I3). The policy
# dispatches to it; the only thing that changes across tiers is which callable runs underneath.
_SUBSTRATES: dict = {}          # Tier -> Callable matching run_program


def register_substrate(tier: Tier, runner: Callable) -> None:
    """Register a real runner for a tier (called by sandbox_linux.py / sandbox_container.py when present).

    `runner` MUST match the frozen run_program signature exactly (invariant I3):
        runner(program, X_train, y_train, X_eval, *, kind, wall_seconds, cpu_seconds, address_mb) -> RunResult
    so no caller above the sandbox changes when the tier changes. We do not validate the signature at
    registration (duck-typed dispatch), but the contract is the registration's responsibility.
    """
    if not callable(runner):
        raise TypeError(f"substrate for {tier} must be callable")
    _SUBSTRATES[tier] = runner


def t2_status(caps: HostCapabilities) -> dict:
    """Honest Tier-2 availability report for THIS host. Used by resolve() and by the audit log.

    Tier 2 (Linux uid + no-new-privs + seccomp + netns + uid-confined fs) requires `setpriv`. Without it
    the whole tier is unavailable. seccomp / netns are sub-capabilities that further degrade the tier's
    network guarantee even when setpriv IS present (design 05 section 3.3 R4)."""
    available = caps.has_setpriv and caps.posix and caps.platform == "linux"
    if available:
        if caps.has_unshare:
            network = "netns"
        elif caps.has_seccomp:
            network = "seccomp-socket-block"
        else:
            network = "not-isolated"            # honest: no netns, no seccomp -> egress open (R4)
        seccomp = "profile" if caps.has_seccomp else "unavailable"
    else:
        network = "not-isolated"
        seccomp = "unavailable"
    reason = ("available" if available
              else f"setpriv/linux unavailable on {caps.platform} (setpriv={caps.has_setpriv})")
    registered = Tier.LINUX_UID in _SUBSTRATES
    return {"available": bool(available), "substrate_registered": registered,
            "network": network, "seccomp": seccomp, "reason": reason}


def t3_status(caps: HostCapabilities, *, gpu: bool = False) -> dict:
    """Honest Tier-3 availability report. Tier 3 needs a container runtime; the GPU lane also needs a GPU.

    NOTE: capability presence (docker on PATH) is necessary but NOT sufficient to claim T3 ran -- a real
    container launch + RO-FS + cgroup + --network none must succeed, which is the sandbox_container.py
    substrate's job. Until that substrate is registered, T3 is "contract-only" here and resolve() degrades.
    """
    runtime = caps.has_docker
    available = runtime and (caps.has_gpu if gpu else True)
    tier = Tier.POD_GPU if gpu else Tier.CONTAINER
    registered = tier in _SUBSTRATES
    if gpu and runtime and not caps.has_gpu:
        reason = "container runtime present but no usable GPU (nvidia-smi absent/failed)"
    elif not runtime:
        reason = "no container runtime (docker/podman) on PATH"
    else:
        reason = "available"
    return {"available": bool(available), "substrate_registered": registered,
            "runtime": bool(runtime), "gpu": bool(caps.has_gpu), "reason": reason}


# ===================================================================== SandboxPolicy
class StrictRefusal(RuntimeError):
    """Raised when strict=True and the policy would otherwise run maximally-untrusted code on LOCAL
    (resource-only isolation). Lets CI for the CORE require T2+ for untrusted code (design 05 H3)."""


class SandboxPolicy:
    """Select a tier by trust + availability, dispatch through the right substrate, and stamp every
    result with the guarantees that ACTUALLY held (probe-derived, never the requested tier).

    The runner contract is unchanged: `run(...)` takes and returns exactly what the Phase-0
    `run_program(...)` does. Tier differences are substrate swaps underneath one fixed signature (I3).

    Construction:
      requested      : the tier the caller WANTS. resolve() caps the result at this and degrades downward.
      untrusted      : True if the code is maximally-untrusted (LLM-authored). Governs the H3 LOCAL warning.
      strict         : if True, REFUSE (raise StrictRefusal) rather than run untrusted code on LOCAL.
      local_runner   : the Tier-1 substrate (inject `frontier.sandbox.run_program`; we never import the
                       spine sandbox here to keep this module importable without it and to make the
                       dependency explicit and swappable in tests). REQUIRED to actually run.
      ast_gate       : run the advisory AST triage before dispatch (default True).
      ast_checker    : a custom AstPolicyChecker (e.g. torch-extended allow-list); default the module one.
      caps           : pre-probed HostCapabilities (default: probe at construction).
    """

    def __init__(self, requested: Tier = Tier.LOCAL, *, untrusted: bool = True, strict: bool = False,
                 local_runner: Optional[Callable] = None, ast_gate: bool = True,
                 ast_checker: Optional[AstPolicyChecker] = None,
                 caps: Optional[HostCapabilities] = None):
        self.requested = Tier(requested)
        self.untrusted = bool(untrusted)
        self.strict = bool(strict)
        self.local_runner = local_runner
        self.ast_gate = bool(ast_gate)
        self.ast_checker = ast_checker or _DEFAULT_AST
        self.caps = caps or probe_host()

    # ----------------------------------------------------------------- tier resolution (downward only)
    def resolve(self) -> tuple:
        """Pick the STRONGEST tier that is AVAILABLE on this host, capped by `requested`. DOWNWARD ONLY.

        Returns (resolved_tier, notes) where notes is an ordered list of honest degradation strings (the
        empty list when the requested tier ran as asked). The resolution NEVER upgrades past `requested`
        and NEVER claims a tier whose substrate is unavailable/unregistered (H2). When a real substrate
        for the requested tier is registered AND its capabilities are present, that tier is kept; otherwise
        we walk down toward LOCAL, recording why at each step.
        """
        notes: list = []
        target = self.requested

        # Walk down from the requested tier until we hit one that is genuinely available on this host.
        order = [Tier.POD_GPU, Tier.CONTAINER, Tier.LINUX_UID, Tier.LOCAL]
        # only consider tiers <= requested
        candidates = [t for t in order if _RANK[t] <= _RANK[target]]
        resolved = Tier.LOCAL
        for t in candidates:
            if t == Tier.LOCAL:
                resolved = Tier.LOCAL
                break
            if t in (Tier.CONTAINER, Tier.POD_GPU):
                st = t3_status(self.caps, gpu=(t == Tier.POD_GPU))
            elif t == Tier.LINUX_UID:
                st = t2_status(self.caps)
            else:
                continue
            if st["available"] and st["substrate_registered"]:
                resolved = t
                break
            # Build an honest "why not" that distinguishes a missing host capability from a present
            # capability with no registered substrate (the design's T2/T3 substrate not yet wired here).
            if not st["available"]:
                why = st["reason"]
            elif not st["substrate_registered"]:
                why = "host capable but no substrate registered (T2/T3 module not wired)"
            else:
                why = "unavailable"
            notes.append(f"requested {target.value}: {t.value} -> {why}; degrade")

        if resolved != target:
            _LOG.warning("sandbox tier degraded: requested=%s resolved=%s; %s",
                         target.value, resolved.value, "; ".join(notes) or "no reason")

        # H3: maximally-untrusted code on LOCAL = resource-only isolation. Warn loudly; refuse if strict.
        if resolved == Tier.LOCAL and self.untrusted:
            msg = ("UNTRUSTED code resolving to Tier-1 LOCAL: resource + crash isolation only, "
                   "NO network isolation, NO memory cap on macOS, NO containment against a C-ext escape. "
                   "Move untrusted code to Tier 2/3 (a Linux uid / no-network container).")
            if self.strict:
                raise StrictRefusal(msg + " [strict mode: refusing to run]")
            _LOG.warning(msg)
            notes.append("untrusted-on-LOCAL: resource-only isolation (loud warning emitted)")

        return resolved, notes

    # ----------------------------------------------------------------- enforced-guarantees probe
    def _probe_enforced(self, tier: Tier, notes: list, *, ast_ran: bool) -> EnforcedGuarantees:
        """Build the EnforcedGuarantees for `tier` from probed capabilities (H1). The values reflect what
        THIS host actually enforces for the resolved tier -- never the requested tier."""
        caps = self.caps
        wall = True                                      # the parent wall-clock backstop is always active
        cpu = caps.rlimit_cpu                            # RLIMIT_CPU set by the substrate's preexec

        if tier == Tier.LOCAL:
            mem = "rlimit-as" if caps.rlimit_as_enforced else "wall-timeout-only"
            return EnforcedGuarantees(
                tier=tier, cpu_rlimit=cpu, mem_guard=mem, wall_timeout=wall,
                network="not-isolated", fs="tempdir-cwd", uid_drop=False, no_new_privs=False,
                seccomp="unavailable",
                fork_guard="pgkill",                     # process-group kill (start_new_session) only
                advisory_ast_gate=ast_ran, notes=tuple(notes))

        if tier == Tier.LINUX_UID:
            st = t2_status(caps)
            mem = "rlimit-as" if caps.rlimit_as_enforced else "wall-timeout-only"
            fork = "nproc+pgkill" if caps.rlimit_nproc else "pgkill"
            return EnforcedGuarantees(
                tier=tier, cpu_rlimit=cpu, mem_guard=mem, wall_timeout=wall,
                network=st["network"], fs="uid-confined", uid_drop=True, no_new_privs=True,
                seccomp=st["seccomp"], fork_guard=fork + "+uid-reap",
                advisory_ast_gate=ast_ran, notes=tuple(notes))

        # CONTAINER / POD_GPU: the container runtime enforces cgroup mem, no-network, RO-FS, pids cgroup;
        # the inner runner still does uid-drop + rlimits + result guards (two-jail nesting, innovation N3).
        gpu = tier == Tier.POD_GPU
        return EnforcedGuarantees(
            tier=tier, cpu_rlimit=cpu, mem_guard="cgroup+rlimit", wall_timeout=wall,
            network="none", fs="readonly-root+scratch", uid_drop=True, no_new_privs=True,
            seccomp="gpu-profile" if gpu else "profile",
            fork_guard="nproc+pidcg+uid-reap",
            advisory_ast_gate=ast_ran, notes=tuple(notes))

    # ----------------------------------------------------------------- the front door
    def run(self, program: Program, X_train, y_train, X_eval, *, kind: str,
            wall_seconds: float = 60.0, cpu_seconds: int = 55, address_mb: int = 4096) -> RunResult:
        """Resolve a tier, optionally run the advisory AST triage, dispatch through the resolved
        substrate, and STAMP the result with the guarantees that actually held.

        Returns the SAME frontier.program.RunResult the substrate returns, with `enforced`
        (an EnforcedGuarantees) attached as a dynamic attribute (RunResult is a frozen Phase-0 type we
        cannot edit; read it back with `enforced_of(result)`). The firewall is preserved end to end: the
        substrate returns predictions only; this layer never computes a metric.
        """
        resolved, notes = self.resolve()

        # (1) ADVISORY AST triage BEFORE dispatch. On a violation, reject WITHOUT executing -> a cheap,
        #     auditable early reject and a clean error taxonomy. Triage, NOT containment.
        ast_ran = False
        if self.ast_gate:
            rep = self.ast_checker.check(program.code)
            ast_ran = True
            if not rep.ok:
                res = RunResult(program.id, ok=False,
                                error="advisory AST policy: " + "; ".join(rep.violations[:6]),
                                error_kind="policy", wall_seconds=0.0)
                enforced = self._probe_enforced(resolved, notes, ast_ran=ast_ran)
                _attach_enforced(res, enforced)
                _audit_log(program.id, enforced, outcome="policy-reject")
                return res

        # (2) dispatch through the resolved substrate. LOCAL uses the injected Phase-0 runner; T2/T3 use a
        #     registered substrate (same signature, I3). If a higher tier resolved but somehow has no
        #     registered runner (race), fall back to LOCAL and note it (never silently claim the higher tier).
        runner = _SUBSTRATES.get(resolved)
        if runner is None:
            if resolved != Tier.LOCAL:
                notes.append(f"no registered substrate for {resolved.value} at dispatch -> LOCAL")
                resolved = Tier.LOCAL
            runner = self.local_runner
        if runner is None:
            res = RunResult(program.id, ok=False,
                            error="no runner available (inject local_runner=frontier.sandbox.run_program)",
                            error_kind="other", wall_seconds=0.0)
            enforced = self._probe_enforced(resolved, notes, ast_ran=ast_ran)
            _attach_enforced(res, enforced)
            _audit_log(program.id, enforced, outcome="no-runner")
            return res

        res = runner(program, X_train, y_train, X_eval, kind=kind,
                     wall_seconds=wall_seconds, cpu_seconds=cpu_seconds, address_mb=address_mb)

        # (3) stamp + audit. EnforcedGuarantees reflects the substrate that ACTUALLY ran (H1/H2).
        enforced = self._probe_enforced(resolved, notes, ast_ran=ast_ran)
        _attach_enforced(res, enforced)
        _audit_log(program.id, enforced, outcome=("ok" if res.ok else f"err:{res.error_kind}"))
        return res


# ===================================================================== enforced-attr helpers + audit
def _attach_enforced(result: RunResult, enforced: EnforcedGuarantees) -> None:
    """Attach EnforcedGuarantees to a RunResult as a dynamic attribute. RunResult is a frozen Phase-0
    dataclass (not slotted) we are forbidden to edit, so we set an attribute rather than add a field. The
    certificate writer reads it via `enforced_of` and stamps the ENFORCED tier into provenance."""
    try:
        object.__setattr__(result, "enforced", enforced)
    except (AttributeError, TypeError):
        # extremely defensive: if RunResult ever became slotted, fall back to a side dict keyed by id().
        _ENFORCED_SIDE[id(result)] = enforced


_ENFORCED_SIDE: dict = {}


def enforced_of(result: RunResult) -> Optional[EnforcedGuarantees]:
    """Read back the EnforcedGuarantees stamped on a RunResult (None if the result never went through a
    SandboxPolicy). Safe for callers/certificate writers."""
    g = getattr(result, "enforced", None)
    if g is not None:
        return g
    return _ENFORCED_SIDE.get(id(result))


def _audit_log(program_id: str, enforced: EnforcedGuarantees, *, outcome: str) -> None:
    """Emit ONE structured audit line per run recording WHAT ACTUALLY HELD (not what was requested).

    This is the mechanical realization of "honest tiering" (invariant I4): a downstream reader can grep
    the log and see, for every executed candidate, the isolation that was in force. INFO level so it is on
    by default for an autoresearch run but suppressible. The certificate carries the same `enforced` dict.
    """
    e = enforced.as_dict()
    _LOG.info("sandbox-run program=%s outcome=%s tier=%s network=%s mem_guard=%s fs=%s "
              "uid_drop=%s no_new_privs=%s seccomp=%s fork_guard=%s ast_gate=%s",
              program_id, outcome, e["tier"], e["network"], e["mem_guard"], e["fs"],
              e["uid_drop"], e["no_new_privs"], e["seccomp"], e["fork_guard"], e["advisory_ast_gate"])


__all__ = [
    "Tier", "EnforcedGuarantees", "HostCapabilities", "probe_host",
    "AstPolicyChecker", "AstReport", "ast_check",
    "ALLOWED_IMPORT_ROOTS", "DENIED_IMPORT_ROOTS", "BANNED_NAMES", "BANNED_ATTRS",
    "scrub_env", "validate_predictions",
    "SandboxPolicy", "StrictRefusal", "register_substrate", "enforced_of",
    "t2_status", "t3_status",
]
