"""Integrity tests for the author-able harness fabric (Phase 3, P3).

Run with the project interpreter:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_harness_authoring.py

These exercise the accept/reject GATE with deterministic stand-in "authored" harnesses, so the
test runs with NO LLM. The properties asserted:
  - a correct authored harness CERTIFIES on a known-good benchmark and is accepted + registered;
  - a broken authored harness (signal-destroying) is REJECTED with reasons and NOT registered;
  - the firewall holds: the registry never hands out an uncertified adapter;
  - honest degradation: llm_client=None with no stand-in => status="inactive", not fabricated;
  - a NON-TABULAR (text) harness runs end to end and certifies through the same frozen gate
    (the ROADMAP Phase-3 acceptance shape);
  - an accepted harness's to_task builds a usable Task on NEW data of that task type.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from frontier.engine import EngineConfig, ResearchEngine


def _load_authoring():
    """Import frontier.harness.authoring robustly.

    The frontier.harness package __init__.py is owned by a SIBLING Phase-3 module (the
    self-testing harness fabric: base/tabular/text/router) authored in parallel. This module
    (authoring.py) has NO dependency on those files -- it imports only from the frontier spine
    (sandbox/engine/task/program). To keep this test independent of the sibling's package
    __init__ (which may be mid-write under concurrent authoring), we load authoring.py directly
    by file path under a lightweight stub package, falling back to that path if the normal
    package import fails to compile. Either way we exercise the SAME authoring.py on disk.
    """
    try:
        return importlib.import_module("frontier.harness.authoring")
    except Exception:
        pass
    # stub the subpackage so relative imports (from .. import sandbox) resolve to the real spine
    pkg_name = "frontier.harness"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [os.path.join(_REPO_ROOT, "frontier", "harness")]
        sys.modules[pkg_name] = pkg
    path = os.path.join(_REPO_ROOT, "frontier", "harness", "authoring.py")
    spec = importlib.util.spec_from_file_location("frontier.harness.authoring", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frontier.harness.authoring"] = mod
    spec.loader.exec_module(mod)
    return mod


_authoring = _load_authoring()
BROKEN_CONSTANT_HARNESS = _authoring.BROKEN_CONSTANT_HARNESS
REFERENCE_TABULAR_HARNESS = _authoring.REFERENCE_TABULAR_HARNESS
REFERENCE_TEXT_HARNESS = _authoring.REFERENCE_TEXT_HARNESS
AuthoredHarness = _authoring.AuthoredHarness
HarnessRegistry = _authoring.HarnessRegistry
KnownGoodBenchmark = _authoring.KnownGoodBenchmark
author_harness = _authoring.author_harness

# Fast self-test config: one round is enough since the reference seeds clear theta.
_CFG = EngineConfig(rounds=1, wall_seconds=60, cpu_seconds=50)


def test_inactive_without_author():
    """No llm_client and no stand-in => inactive, accepted=False, no fabrication."""
    bench = KnownGoodBenchmark.builtin_tabular()
    a = author_harness("tabular", bench, llm_client=None, config=_CFG)
    assert a.status == "inactive", a.status
    assert a.accepted is False
    assert a.self_test_certificate is None
    assert any("inactive" in r for r in a.reasons)
    print(f"[ok] inactive without author: {a.reasons[0][:70]}")


def test_good_tabular_harness_accepted_and_registered():
    """A correct authored tabular harness certifies on the known-good benchmark and registers."""
    reg = HarnessRegistry()
    bench = KnownGoodBenchmark.builtin_tabular()
    a = reg.author_and_register("tabular_clf", bench, llm_client=None,
                                _authored_code=REFERENCE_TABULAR_HARNESS, config=_CFG)
    assert a.accepted, f"good harness should be accepted; reasons={a.reasons}"
    assert a.status == "accepted"
    c = a.self_test_certificate
    assert c is not None and c.get("certified") is True
    assert c["peeks"] == 1, "self-test must touch sealed exactly once"
    assert c["lower_bound"] > c["theta"], "certified means sealed lower bound clears theta"
    assert reg.has("tabular_clf") and reg.get("tabular_clf") is a
    print(f"[ok] tabular accepted+registered: lb={c['lower_bound']} > theta={c['theta']} "
          f"peeks={c['peeks']}")


def test_broken_harness_rejected_and_not_registered():
    """A signal-destroying harness cannot certify and is rejected with a reason; not registered."""
    reg = HarnessRegistry()
    bench = KnownGoodBenchmark.builtin_tabular()
    a = reg.author_and_register("tabular_clf", bench, llm_client=None,
                                _authored_code=BROKEN_CONSTANT_HARNESS, config=_CFG)
    assert not a.accepted, "constant-feature harness must NOT be accepted"
    assert a.status == "rejected"
    assert a.reasons and "did not certify" in a.reasons[0]
    assert not reg.has("tabular_clf"), "rejected harness must not enter the registry"
    print(f"[ok] broken rejected (not registered): {a.reasons[0][:80]}")


def test_registry_never_hands_out_uncertified():
    """The trust boundary: after a rejected author attempt, get() returns None."""
    reg = HarnessRegistry()
    bench = KnownGoodBenchmark.builtin_tabular()
    reg.author_and_register("t", bench, _authored_code=BROKEN_CONSTANT_HARNESS, config=_CFG)
    assert reg.get("t") is None
    assert reg.task_types() == []
    print("[ok] registry refuses uncertified adapters")


def test_malformed_harness_rejected_with_reason():
    """A module missing build_task is rejected pre-sandbox with a clear reason."""
    bench = KnownGoodBenchmark.builtin_tabular()
    a = author_harness("t", bench, _authored_code="def not_it(raw):\n    return {}\n", config=_CFG)
    assert not a.accepted and a.status == "rejected"
    assert any("build_task" in r for r in a.reasons)
    print(f"[ok] malformed rejected: {a.reasons[0][:70]}")


def test_nontabular_text_harness_end_to_end():
    """Phase-3 acceptance: a NON-TABULAR (text) task type runs end to end and certifies."""
    reg = HarnessRegistry()
    bench = KnownGoodBenchmark.builtin_text()
    a = reg.author_and_register("short_text_topic", bench, llm_client=None,
                                _authored_code=REFERENCE_TEXT_HARNESS, config=_CFG)
    assert a.accepted, f"text harness should certify; reasons={a.reasons}"
    c = a.self_test_certificate
    assert c is not None and c["certified"] and c["peeks"] == 1
    assert reg.has("short_text_topic")
    print(f"[ok] non-tabular text certified e2e: lb={c['lower_bound']} > theta={c['theta']}")


def test_accepted_harness_builds_task_on_new_data():
    """An accepted text harness produces a usable Task for NEW data of the same task type."""
    bench = KnownGoodBenchmark.builtin_text()
    a = author_harness("short_text_topic", bench, _authored_code=REFERENCE_TEXT_HARNESS, config=_CFG)
    assert a.accepted
    # brand-new raw data of the same task type (disjoint vocab classes)
    new_texts = ["puck rink goalie the and"] * 8 + ["orbit rocket galaxy of to"] * 8
    new_labels = ["hockey"] * 8 + ["space"] * 8
    task = a.to_task((new_texts, new_labels))
    assert task.kind == "classification"
    assert task.X.ndim == 2 and task.X.shape[0] == 16
    assert len(task.y) == 16 and set(task.y) == {"hockey", "space"}
    print(f"[ok] accepted harness built new Task: X{task.X.shape} labels={sorted(set(task.y))}")


def test_unaccepted_to_task_refuses():
    """to_task on an unaccepted harness raises (no untrusted adapter used on real data)."""
    a = AuthoredHarness(task_type="t", code="", status="rejected", accepted=False,
                        reasons=["test"])
    raised = False
    try:
        a.to_task({"data": [[0.0]], "target": ["a"]})
    except RuntimeError:
        raised = True
    assert raised, "to_task must refuse on an unaccepted harness"
    print("[ok] unaccepted to_task refuses")


def test_llm_client_path_with_fake_client():
    """The llm_client path is exercised with a deterministic fake model returning correct code."""
    bench = KnownGoodBenchmark.builtin_tabular()

    def fake_client(prompt: str) -> str:
        # a 'frontier model' that returns a correct tabular harness, with stray fences to
        # prove the fence-stripper works.
        return "```python\n" + REFERENCE_TABULAR_HARNESS + "```"

    a = author_harness("tabular_via_llm", bench, llm_client=fake_client, config=_CFG)
    assert a.accepted, f"fake-LLM-authored harness should certify; reasons={a.reasons}"
    assert a.provenance.get("author") == "llm"
    print(f"[ok] llm_client path accepted (fences stripped): {a.status}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} harness-authoring tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
