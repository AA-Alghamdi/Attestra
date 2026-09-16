"""Containment proof for the OS-level isolation runner (NEW module vfplatform/authored_sandbox.py).

These tests drive `run_in_sandbox(...)` THROUGH the runner and assert that the five CONFIRMED
attack vectors from the adversarial pass are contained, while a legitimate authored estimator's
fit/predict returns correct predictions ACROSS the process boundary.

  CONTAINED (each must FAIL / leak nothing):
    1. np.fromfile + np.genfromtxt reading vectorforge/.anthropic_key  -> PermissionError (sandbox-exec)
    2. np.savetxt writing a host file outside the jail                 -> PermissionError (sandbox-exec)
    3. numpy mem-bomb                                                  -> killed by the RSS watchdog (rc -9)
    4. CPU runaway                                                     -> killed by RLIMIT_CPU (rc 152) / wall
    5. socket network egress                                          -> PermissionError (sandbox-exec)

  CORRECT ACROSS THE BOUNDARY:
    - a benign OLS / RandomForest estimator fits + predicts in the child and returns predictions
      that match an in-process reference fit (the predictions really crossed the JSON file boundary).

On a host WITHOUT sandbox-exec (e.g. Linux dev w/o container) the filesystem/network OS controls are
absent; those four fs/net assertions are skipped with a recorded reason (containment there comes from
the container boundary, per the module docstring). The CPU/mem/wall caps and the legit round-trip are
asserted on every host. Frozen hashes are re-verified at the end.

Run:  PYTHONPATH=. /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_authored_sandbox.py
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import authored_sandbox as A   # noqa: E402

_SECRET_DIR = A._SECRET_DIR
_KEY_PATH = os.path.join(_SECRET_DIR, ".anthropic_key")
_HAVE_SBEXEC = (sys.platform == "darwin") and os.path.exists(A._SANDBOX_EXEC)
_EXFIL_PATHS = ("/tmp/vf_exfil_attempt.txt", "/private/tmp/vf_exfil_attempt.txt",
                "/tmp/vf_test_exfil.txt", "/private/tmp/vf_test_exfil.txt")

# Frozen, untouchable modules (must remain byte-identical: this module is leaf-only).
_FROZEN = {
    "vectorforge/science.py": "b564fba248ea009495fcfc2cb7e2e14a1d1285c2f46165cd9401dfbcd461270d",
    "vfplatform/sealed.py": "30ad62450c53cc4a52ed55e65593352c146549d24e756dcb38532569007c8661",
}
_REPO = Path(__file__).resolve().parent.parent


def _data():
    rng = np.random.RandomState(0)
    Xtr = rng.randn(60, 3)
    ytr = Xtr @ np.array([1.0, 2.0, -1.0]) + 0.05 * rng.randn(60)
    Xte = rng.randn(10, 3)
    return Xtr, ytr, Xte


# ============================================================ legitimate estimator across the boundary
_BENIGN_OLS = '''
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

_BENIGN_RF = '''
import numpy as np
from sklearn.ensemble import RandomForestRegressor
class _RF:
    def fit(self, X, y):
        self.m_ = RandomForestRegressor(n_estimators=25, random_state=0).fit(X, y)
        return self
    def predict(self, X):
        return self.m_.predict(X)
def build_estimator():
    return _RF()
'''


def test_benign_ols_predictions_cross_boundary():
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_BENIGN_OLS, Xtr, ytr, Xte)
    assert r.ok, f"benign OLS failed in sandbox: {r.reason}"
    assert r.predictions is not None and len(r.predictions) == Xte.shape[0], \
        f"expected {Xte.shape[0]} predictions, got {None if r.predictions is None else len(r.predictions)}"
    # the predictions must MATCH a reference in-process fit -> they really crossed the boundary correctly
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    ref = Xte @ beta
    got = np.asarray(r.predictions, dtype=float)
    assert np.allclose(got, ref, atol=1e-6), \
        f"sandbox predictions differ from reference (max abs diff {np.max(np.abs(got - ref)):.2e})"
    return "benign OLS fit/predict crossed the boundary and matched the in-process reference"


def test_benign_rf_predictions_cross_boundary():
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_BENIGN_RF, Xtr, ytr, Xte)
    assert r.ok, f"benign RandomForest failed in sandbox: {r.reason}"
    assert r.predictions is not None and len(r.predictions) == Xte.shape[0]
    assert all(np.isfinite(r.predictions)), "non-finite predictions across the boundary"
    return f"benign RF (sklearn import allowed) returned {len(r.predictions)} finite predictions across the boundary"


# ============================================================ Attack 1: secret-file read (fromfile + genfromtxt)
def _attack_fromfile_src():
    return (
        "import numpy as np\n"
        f"KEY = {_KEY_PATH!r}\n"
        "class _Steal:\n"
        "    def fit(self, X, y):\n"
        "        self.leak_ = np.fromfile(KEY, dtype=np.uint8)\n"
        "        self.n_ = float(self.leak_.sum())\n"  # encode stolen bytes into the predictions
        "        return self\n"
        "    def predict(self, X):\n"
        "        return np.full(X.shape[0], self.n_)\n"
        "def build_estimator():\n"
        "    return _Steal()\n"
    )


def _attack_genfromtxt_src():
    return (
        "import numpy as np\n"
        f"KEY = {_KEY_PATH!r}\n"
        "class _Steal:\n"
        "    def fit(self, X, y):\n"
        "        self.leak_ = np.genfromtxt(KEY, dtype='U200', delimiter='\\n')\n"
        "        return self\n"
        "    def predict(self, X):\n"
        "        return np.zeros(X.shape[0])\n"
        "def build_estimator():\n"
        "    return _Steal()\n"
    )


def test_attack_fromfile_secret_read_contained():
    if not _HAVE_SBEXEC:
        return "SKIP (no sandbox-exec; fs isolation relies on the container boundary in prod)"
    assert os.path.exists(_KEY_PATH), "precondition: the secret key must exist for the test to be meaningful"
    key_before = Path(_KEY_PATH).read_bytes()
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_attack_fromfile_src(), Xtr, ytr, Xte)
    assert not r.ok, f"np.fromfile of the secret was NOT contained: {r.reason!r}"
    assert "PermissionError" in r.reason or "Operation not permitted" in r.reason, \
        f"expected a permission denial, got {r.reason!r}"
    # nothing readable crossed the boundary
    assert r.predictions is None, f"predictions leaked across the boundary: {r.predictions!r}"
    assert Path(_KEY_PATH).read_bytes() == key_before, "the secret key was modified"
    return f"np.fromfile(.anthropic_key) blocked by sandbox-exec ({r.reason[:60]}...)"


def test_attack_genfromtxt_secret_read_contained():
    if not _HAVE_SBEXEC:
        return "SKIP (no sandbox-exec; fs isolation relies on the container boundary in prod)"
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_attack_genfromtxt_src(), Xtr, ytr, Xte)
    assert not r.ok, f"np.genfromtxt of the secret was NOT contained: {r.reason!r}"
    assert "PermissionError" in r.reason or "Operation not permitted" in r.reason \
        or "error" in r.reason.lower(), f"unexpected reason {r.reason!r}"
    assert r.predictions is None, f"predictions leaked across the boundary: {r.predictions!r}"
    return f"np.genfromtxt(.anthropic_key) blocked ({r.reason[:60]}...)"


# ============================================================ Attack 2: write a host file (savetxt)
_ATK_WRITE = '''
import numpy as np
class _Dump:
    def fit(self, X, y):
        np.savetxt("/tmp/vf_test_exfil.txt", X)
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
def build_estimator():
    return _Dump()
'''


def test_attack_savetxt_host_write_contained():
    for p in _EXFIL_PATHS:
        try:
            os.remove(p)
        except OSError:
            pass
    if not _HAVE_SBEXEC:
        return "SKIP (no sandbox-exec; fs write isolation relies on the container boundary in prod)"
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_ATK_WRITE, Xtr, ytr, Xte)
    assert not r.ok, f"np.savetxt host write was NOT contained: {r.reason!r}"
    assert "PermissionError" in r.reason or "Operation not permitted" in r.reason, \
        f"expected a permission denial, got {r.reason!r}"
    for p in _EXFIL_PATHS:
        assert not os.path.exists(p), f"exfil file {p} was actually written"
    return f"np.savetxt outside the jail blocked; no host file written ({r.reason[:60]}...)"


# ============================================================ Attack 3: memory bomb
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


def test_attack_membomb_killed_by_cap():
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_ATK_MEMBOMB, Xtr, ytr, Xte, mem_mb=256, cpu_s=20, wall_s=40)
    assert not r.ok, "mem-bomb was NOT contained"
    # On Darwin: RSS watchdog SIGKILL (rc -9, reason mentions memory). On Linux: RLIMIT_AS MemoryError.
    killed = (r.returncode in (-9,)) or ("memory" in r.reason.lower()) or ("MemoryError" in r.reason)
    assert killed, f"mem-bomb not killed by the cap: rc={r.returncode} reason={r.reason!r}"
    assert r.predictions is None
    return f"mem-bomb killed (rc={r.returncode}, peakRSS={r.peak_rss_mb:.0f}MB, reason={r.reason[:50]!r})"


# ============================================================ Attack 4: CPU runaway
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


def test_attack_cpubomb_killed_by_cap():
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_ATK_CPUBOMB, Xtr, ytr, Xte, mem_mb=1024, cpu_s=2, wall_s=20)
    assert not r.ok, "cpu-bomb was NOT contained"
    # RLIMIT_CPU SIGXCPU -> child emits rc 152; wall-clock kill (rc -9 / reason 'wall') is the backstop.
    killed = (r.returncode in (152, -9)) or ("cpu" in r.reason.lower()) or ("wall" in r.reason.lower())
    assert killed, f"cpu-bomb not killed: rc={r.returncode} reason={r.reason!r}"
    # the CPU cap must fire well before the wall deadline
    assert r.wall_s < 20.0, f"cpu cap did not fire before the wall deadline (wall={r.wall_s:.1f}s)"
    return f"cpu-bomb killed (rc={r.returncode}, wall={r.wall_s:.2f}s, reason={r.reason[:50]!r})"


# ============================================================ Attack 5: network egress
_ATK_NET = '''
import numpy as np, socket
class _Phone:
    def fit(self, X, y):
        socket.create_connection(("1.1.1.1", 443), timeout=3)
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
def build_estimator():
    return _Phone()
'''


def test_attack_network_egress_contained():
    if not _HAVE_SBEXEC:
        return "SKIP (no sandbox-exec; network isolation relies on container --network=none in prod)"
    Xtr, ytr, Xte = _data()
    r = A.run_in_sandbox(_ATK_NET, Xtr, ytr, Xte)
    assert not r.ok, f"network egress was NOT contained: {r.reason!r}"
    assert "PermissionError" in r.reason or "Operation not permitted" in r.reason, \
        f"expected a network denial, got {r.reason!r}"
    return f"socket.create_connection blocked by sandbox-exec ({r.reason[:50]}...)"


# ============================================================ runner hygiene
def test_runner_never_raises_and_cleans_tmp():
    # malformed source must come back as ok=False, not an exception
    r = A.run_in_sandbox("this is not python", *_data())
    assert not r.ok and r.reason, "malformed source should return ok=False with a reason"
    # no vf_authsbx_ jail should leak into TMPDIR (finally: rmtree)
    import tempfile
    leftovers = [p for p in os.listdir(tempfile.gettempdir()) if p.startswith("vf_authsbx_")]
    assert not leftovers, f"sandbox tmpdirs were not cleaned: {leftovers}"
    return "runner never raises and cleans its per-run tmpdir"


# ============================================================ leaf-only + frozen integrity
def test_module_is_leaf_only():
    src = (_REPO / "vfplatform" / "authored_sandbox.py").read_text()
    for forbidden in ("import loop", "from .loop", "from vfplatform.loop",
                      "import sealed", "from .sealed", "science.py"):
        assert forbidden not in src, f"authored_sandbox.py must not reference {forbidden!r}"
    return "authored_sandbox.py is a leaf module (no loop/sealed/science execution imports)"


def test_frozen_hashes_unchanged():
    for rel, want in _FROZEN.items():
        got = hashlib.sha256((_REPO / rel).read_bytes()).hexdigest()
        assert got == want, f"FROZEN FILE CHANGED: {rel}\n  want {want}\n  got  {got}"
    return f"frozen hashes intact ({', '.join(_FROZEN)})"


TESTS = [
    test_benign_ols_predictions_cross_boundary,
    test_benign_rf_predictions_cross_boundary,
    test_attack_fromfile_secret_read_contained,
    test_attack_genfromtxt_secret_read_contained,
    test_attack_savetxt_host_write_contained,
    test_attack_membomb_killed_by_cap,
    test_attack_cpubomb_killed_by_cap,
    test_attack_network_egress_contained,
    test_runner_never_raises_and_cleans_tmp,
    test_module_is_leaf_only,
    test_frozen_hashes_unchanged,
]


def main():
    print(f"host: {sys.platform}  sandbox-exec: {_HAVE_SBEXEC}  python: {sys.executable}")
    passed = failed = 0
    for t in TESTS:
        try:
            msg = t()
            print(f"  PASS  {t.__name__}: {msg}")
            passed += 1
        except Exception as ex:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(ex).__name__}: {ex}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
