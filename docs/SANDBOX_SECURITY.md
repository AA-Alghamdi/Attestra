# Authored-code sandbox - threat model & verification

`vfplatform/authored_pod_sandbox.py` runs **untrusted LLM-authored estimator code** (Wave-3 generative
PROPOSE) on the RunPod A100 pod. RunPod's container blocks the usual isolation primitives - no Docker-in-pod,
no unprivileged user namespaces (`bwrap`/`unshare` fail), no installable seccomp, and `/proc` cannot be
remounted (`hidepid=2` → EPERM). So isolation is built from what the kernel *does* enforce here: uid drop to
dedicated unprivileged pool users, secrets on the local overlay fs (where Unix perms are enforced), a scrubbed
environment, rlimits, reap-by-uid, and a confused-deputy-resistant result reader.

## Design

- **Dedicated per-run uid.** Each run claims a pool user `sbx1..sbx8` (uids 60001-60008, gid `sbx`) via an
  flock slot, and runs the child as `setpriv --reuid sbxN --regid sbx --clear-groups --no-new-privs
  --inh-caps=-all --ambient-caps=-all python3 driver.py`. Pool uids must be **inside the userns uid_map
  range** (0–65535 here); out-of-range uids fail `setresuid` with EINVAL. A dedicated uid (vs shared
  `nobody`) lets us reap precisely and isolates concurrent runs from each other.
- **Secrets on local fs.** The Anthropic key lives at `/root/.attestera/env` (root, mode 600) on the local
  overlay. The `/workspace` volume is a **network fs that ignores `chmod`** (777 at every level) - no secret
  may ever live there. The frozen certifier (`vectorforge/science.py`, `vfplatform/sealed.py`) and the loop
  run from `/root/vectorforge-product` (local, root-owned) so a sandboxed uid cannot tamper with the promoter.
- **Scrubbed env.** Only `PATH,HOME,TMPDIR,VF_*` and thread/arena caps. `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`
  + `MALLOC_ARENA_MAX=2` are **mandatory** on this 256-core box: without them numpy/scipy spawn a thread +
  malloc arena per core at import, which burns RLIMIT_CPU before the child runs and reserves huge virtual AS.
- **rlimits** (set in the child before importing user code): `RLIMIT_AS` (mem - generous enough to import
  numpy/sklearn yet catch real bombs), `RLIMIT_CPU`, `RLIMIT_FSIZE=64MB`, `RLIMIT_NPROC=64` (fork-bomb),
  `RLIMIT_NOFILE` bounded. Plus a parent wall-clock `subprocess` timeout that SIGKILLs independently.
- **Reap-by-uid loop.** After the child returns/times out (and again in `finally`), `pkill -9 -u sbxN` runs in
  a loop until no process of that uid remains - kills detached `setsid`/`posix_spawn` daemons and
  pdeathsig-reparented grandchildren that escape the parent's process group, and defeats fork-races. Then
  `find /tmp /var/tmp /dev/shm -uid sbxN -delete` removes beacons written outside the scratch dir.
- **Confused-deputy-resistant result read** (`_read_result`): the root parent reads `result.json` with
  `O_NOFOLLOW|O_NONBLOCK`, requires `S_ISREG`, `st_uid==sbxN`, `st_nlink==1`, size ≤ 64MB, parses JSON with
  `parse_constant` rejecting `Infinity`/`NaN`, and requires `pred` to be a flat list of finite floats (bools
  rejected). Architecturally, the trusted driver also writes `result.json` *last*, so the child cannot inject
  content on the success path; the reader guards cover the residual paths (symlink/fifo/hardlink).
- Only **train arrays + an eval-feature matrix** cross the boundary as `.npy`. The **sealed test is never
  passed in**. The authored estimator object never crosses back - only the validated `pred` vector does.

## Verified blocked (all tested on the pod, each failing for the *right* reason)

Secret exfiltration: `open('/root/.attestera/env')` (EACCES), `os.environ['ANTHROPIC_API_KEY']` (KeyError),
`/proc/<ppid>/environ`, `/proc/<ppid>/task/<tid>/environ`, scan-all `/proc/*/environ` (owner-only),
`/proc/<ppid>/mem` heap-scrape (EACCES), `/proc/<ppid>/fd/*` (EACCES), `/proc/self/root` & `/proc/<ppid>/root`
traversal, `/etc/shadow`, hardlink-to-secret (EACCES; `protected_hardlinks`), symlink `result.json`→secret
(O_NOFOLLOW). Privilege: `os.setuid(0)`/`setresuid(0)` (EPERM, no caps in userns), `ptrace(ATTACH)` (denied),
SUID/PATH-hijack (`--no-new-privs`). Kernel/host: `/dev/mem`, `/proc/kcore`, `/proc/sysrq-trigger` (RO fs).
Tamper: write `/root`, append the frozen certifier (EACCES on local fs). Resource: 16GB mem-bomb (RLIMIT_AS),
1GB disk + 50M-elem pred write (RLIMIT_FSIZE), fork-bomb (RLIMIT_NPROC), fd-exhaustion (RLIMIT_NOFILE),
8-thread CPU pin + blocked-pipe (wall timeout), zombie storm, inode/deep-dir/readonly-subdir (rmtree as root).
Persistence: detached `setsid`/`posix_spawn`/`os.fork` daemons, pdeathsig reparenting, atexit-fork - **all
reaped** (no survivors); out-of-scratch beacons purged. Output/deputy: fifo (timeout), root-owned/hardlink/
oversized/Infinity/NaN/nested/bool/string/missing `result.json` - all rejected by `_read_result` (12/12 unit
tests pass).

## Residuals (documented, accepted)

1. **`/workspace` is writable** by any uid (network volume ignores perms; cannot be `chmod`'d or
   mount-namespaced away here). Mitigation: it holds only re-downloadable, **non-secret** data; all critical
   assets (secrets, frozen certifier, loop code, sealed test) are on local fs and verified unwritable. The
   loop must **never put `/workspace` on `sys.path`** or run with `cwd=/workspace`, else a planted
   `sitecustomize.py` could execute (verified: the child's `sys.path` does not include `/workspace`).
2. **`/proc/<pid>/cmdline` is world-readable** (`hidepid` can't be set in this container). A sandboxed child
   *can* read other processes' argv. Mitigation = **hard operational rule: no secret on any argv, ever** - the
   key is read from the local file into process memory (env/var), never passed as a CLI argument. Verified:
   the real key prefix appears on no process's cmdline. (`/proc/<pid>/environ` and `/mem` are owner-only, so
   the in-memory key is not reachable.)
3. **Network egress is open** (no seccomp/iptables-owner). Accepted because there is nothing sensitive to
   exfil: secrets unreadable, env scrubbed, only non-secret train arrays passed in. Close fully with a seccomp
   socket-filter if `libseccomp` becomes installable.

## Not yet wired

The sandbox is **not** wired into `loop.py`. An authored model that runs cleanly is only a *candidate*; the
**frozen certifier** (unchanged, hash-pinned `b564fba2` / `30ad6245`) is the sole promoter. Wiring is Wave-3.
