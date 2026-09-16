"""Tier-2 (LINUX_UID) substrate: real uid + network isolation on a capable Linux host.

These tests exercise the REAL kernel namespace substrate (frontier/core/sandbox_linux.py), not a fake.
They are skipped automatically on any host that cannot create an unprivileged user+network namespace
(macOS, hardened kernels, CI without userns) so the suite stays green everywhere -- the substrate's whole
contract is honest degradation, and skipping where it cannot run is part of that honesty.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontier import sandbox as spine
from frontier.core import sandbox_linux as sl
from frontier.core import sandbox_policy as sp
from frontier.program import Program

pytestmark = pytest.mark.skipif(
    not sl.T2_REGISTERED,
    reason="host cannot create an unprivileged user+network namespace (Tier-2 honestly unavailable)",
)


@pytest.fixture(autouse=True)
def _own_t2_registration():
    # Other suites clear sp._SUBSTRATES for hermiticity; these tests must therefore register the real
    # T2 substrate themselves and restore the prior state after, so they are order-independent.
    saved = dict(sp._SUBSTRATES)
    sp.register_substrate(sp.Tier.LINUX_UID, sl.run_program_t2)
    try:
        yield
    finally:
        sp._SUBSTRATES.clear()
        sp._SUBSTRATES.update(saved)


def test_t2_auto_registers_at_import_and_reports_netns():
    # sl.T2_REGISTERED is set at import time IFF the host could actually create the namespaces -- this
    # proves the real auto-registration path fired (independent of the per-test fixture).
    assert sl.T2_REGISTERED is True, "capable host must auto-register T2 at import"
    assert sp.Tier.LINUX_UID in sp._SUBSTRATES, "substrate must be live for this test"
    st = sp.t2_status(sp.probe_host())
    assert st["available"] is True and st["substrate_registered"] is True
    assert st["network"] == "netns", f"unshare --net must give netns, got {st['network']}"


def test_container_request_resolves_to_t2_not_local():
    # The orchestrator requests CONTAINER; with T3 unregistered but T2 registered, the honest resolve
    # is LINUX_UID -- a REAL upgrade over LOCAL, never a false claim of container isolation.
    pol = sp.SandboxPolicy(sp.Tier.CONTAINER, untrusted=True, strict=False,
                           local_runner=spine.run_program)
    resolved, notes = pol.resolve()
    assert resolved == sp.Tier.LINUX_UID, f"expected T2, got {resolved}"
    assert sp._RANK[resolved] <= sp._RANK[sp.Tier.CONTAINER], "never upgrade past requested"


def test_t2_runs_clean_estimator_and_stamps_isolation():
    pol = sp.SandboxPolicy(sp.Tier.LINUX_UID, untrusted=True, strict=False,
                           local_runner=spine.run_program)
    code = ("from sklearn.linear_model import LogisticRegression\n"
            "def build_estimator():\n    return LogisticRegression(max_iter=200)\n")
    rng = np.random.RandomState(0)
    X = rng.randn(40, 4); y = (X[:, 0] > 0).astype(int).astype(str)
    res = pol.run(Program(code=code, source="seed", label="logreg"),
                  X, y, X[:6], kind="classification")
    assert res.ok, f"clean estimator must run under T2: {res.error}"
    assert len(res.preds) == 6
    g = sp.enforced_of(res)
    assert g.tier == sp.Tier.LINUX_UID
    assert g.uid_drop is True and g.no_new_privs is True
    assert g.network == "netns" and g.fs == "uid-confined"


def test_t2_netns_blocks_egress_even_with_ast_gate_off():
    # Prove the KERNEL (not the AST triage) blocks the network: turn the advisory gate off so a
    # socket-opening candidate reaches execution, and assert it cannot reach the network.
    pol = sp.SandboxPolicy(sp.Tier.LINUX_UID, untrusted=True, strict=False,
                           local_runner=spine.run_program, ast_gate=False)
    rng = np.random.RandomState(1)
    X = rng.randn(20, 3); y = (X[:, 0] > 0).astype(int).astype(str)
    net_code = ("def build_estimator():\n"
                "    import socket\n"
                "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
                "    from sklearn.linear_model import LogisticRegression\n"
                "    return LogisticRegression()\n")
    res = pol.run(Program(code=net_code, source="seed", label="netprobe"),
                  X, y, X[:4], kind="classification")
    assert not res.ok, "network egress must FAIL inside the T2 net namespace"
    assert "unreachable" in res.error.lower() or "network" in res.error.lower(), res.error
    assert sp.enforced_of(res).network == "netns"
