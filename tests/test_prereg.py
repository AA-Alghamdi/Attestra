"""Unit tests for vfplatform.prereg (pre-registration commitment layer).

Plain asserts + main() (no pytest), matching the repo's other tests (e.g.
test_certifier_coverage.py). These unit-test the module in isolation; the loop is NOT
involved (prereg is not wired in yet).

Covers:
  * canonical_spec/commit determinism (same spec -> same hash, key order irrelevant)
  * change-detection: a RELAXED threshold -> a DIFFERENT hash (the spec-bend is visible)
  * any other field change -> different hash
  * PlanRegistry append / count / read / tolerance of missing+corrupt files / atomicity
    (relaxed threshold appended ALONGSIDE original, original preserved)

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_prereg.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vfplatform import prereg


def test_canonical_spec_is_json_and_sorted():
    s = prereg.canonical_spec(metric="balanced_accuracy", threshold=0.80, alpha=0.05,
                              forbidden_fields=("z_field", "a_field", "a_field"))
    import json
    json.dumps(s)  # must be serializable
    # forbidden_fields sorted and de-duplicated
    assert s["forbidden_fields"] == ["a_field", "z_field"]
    assert s["metric"] == "balanced_accuracy"
    assert s["selection_rule"] == "best_val_lower_bound"


def test_commit_determinism_same_spec_same_hash():
    s1 = prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                               forbidden_fields=("a", "b"))
    s2 = prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                               forbidden_fields=("a", "b"))
    assert prereg.commit(s1)["plan_hash"] == prereg.commit(s2)["plan_hash"]


def test_commit_invariant_to_forbidden_field_order():
    s1 = prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                               forbidden_fields=("a", "b", "c"))
    s2 = prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                               forbidden_fields=("c", "b", "a"))
    assert prereg.commit(s1)["plan_hash"] == prereg.commit(s2)["plan_hash"]


def test_commit_invariant_to_float_noise():
    # a YAML round-trip might give 0.8000000001; canonicalization rounds it, so the hash is stable
    s1 = prereg.canonical_spec(metric="r2", threshold=0.8, alpha=0.05)
    s2 = prereg.canonical_spec(metric="r2", threshold=0.8 + 1e-12, alpha=0.05)
    assert prereg.commit(s1)["plan_hash"] == prereg.commit(s2)["plan_hash"]


def test_relaxed_threshold_changes_hash():
    s_strict = prereg.canonical_spec(metric="balanced_accuracy", threshold=0.80, alpha=0.05)
    s_relaxed = prereg.canonical_spec(metric="balanced_accuracy", threshold=0.78, alpha=0.05)
    assert prereg.commit(s_strict)["plan_hash"] != prereg.commit(s_relaxed)["plan_hash"], \
        "relaxing the threshold MUST produce a different hash (visible spec-bend)"


def test_any_field_change_changes_hash():
    base = prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                                 forbidden_fields=("a",), selection_rule="best_val_lower_bound")
    h0 = prereg.commit(base)["plan_hash"]
    variants = [
        prereg.canonical_spec(metric="neg_rmse", threshold=0.5, alpha=0.05,
                              forbidden_fields=("a",)),
        prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.01,
                              forbidden_fields=("a",)),
        prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                              forbidden_fields=("a", "b")),
        prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                              forbidden_fields=("a",), selection_rule="raw_val"),
        prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05,
                              forbidden_fields=("a",), extra={"seed": 7}),
    ]
    for v in variants:
        assert prereg.commit(v)["plan_hash"] != h0, f"field change did not change hash: {v}"


def test_registry_missing_file_is_empty():
    with tempfile.TemporaryDirectory() as d:
        reg = prereg.PlanRegistry(os.path.join(d, "nope.jsonl"))
        assert reg.count() == 0
        assert reg.all() == []


def test_registry_append_count_read():
    with tempfile.TemporaryDirectory() as d:
        reg = prereg.PlanRegistry(os.path.join(d, "plans.jsonl"))
        c1 = prereg.commit(prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05))
        c2 = prereg.commit(prereg.canonical_spec(metric="r2", threshold=0.4, alpha=0.05))
        reg.register(c1)
        reg.register(c2)
        assert reg.count() == 2
        got = reg.all()
        assert got[0]["plan_hash"] == c1["plan_hash"]
        assert got[1]["plan_hash"] == c2["plan_hash"]


def test_relaxed_recorded_alongside_original():
    # The integrity property: relaxing a threshold appends a NEW commitment; the original is
    # still present and unchanged. Both hashes coexist in the file.
    with tempfile.TemporaryDirectory() as d:
        reg = prereg.PlanRegistry(os.path.join(d, "plans.jsonl"))
        strict = prereg.commit(prereg.canonical_spec(metric="balanced_accuracy",
                                                     threshold=0.80, alpha=0.05))
        reg.register(strict)
        relaxed = prereg.commit(prereg.canonical_spec(metric="balanced_accuracy",
                                                      threshold=0.78, alpha=0.05))
        reg.register(relaxed)
        all_h = [c["plan_hash"] for c in reg.all()]
        assert strict["plan_hash"] in all_h, "original commitment must survive"
        assert relaxed["plan_hash"] in all_h, "relaxed commitment must be recorded"
        assert len(set(all_h)) == 2


def test_registry_tolerates_corrupt_line():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "plans.jsonl")
        c1 = prereg.commit(prereg.canonical_spec(metric="r2", threshold=0.5, alpha=0.05))
        reg = prereg.PlanRegistry(path)
        reg.register(c1)
        # inject a corrupt line by appending raw garbage (simulating a torn write)
        with open(path, "a", encoding="utf-8") as f:
            f.write("{not valid json\n")
        # read tolerates the corrupt line (skips it) and still returns the valid one
        assert reg.count() == 1
        assert reg.all()[0]["plan_hash"] == c1["plan_hash"]
        # a subsequent register preserves the valid line and adds a new one
        c2 = prereg.commit(prereg.canonical_spec(metric="r2", threshold=0.4, alpha=0.05))
        reg.register(c2)
        hashes = {c["plan_hash"] for c in reg.all()}
        assert hashes == {c1["plan_hash"], c2["plan_hash"]}


def test_registry_atomic_no_tmp_left_behind():
    with tempfile.TemporaryDirectory() as d:
        reg = prereg.PlanRegistry(os.path.join(d, "plans.jsonl"))
        for i in range(5):
            reg.register(prereg.commit(
                prereg.canonical_spec(metric="r2", threshold=0.5 - 0.01 * i, alpha=0.05)))
        leftover = [f for f in os.listdir(d) if f.startswith(".prereg.")]
        assert leftover == [], f"atomic write left temp files: {leftover}"
        assert reg.count() == 5


def test_bad_inputs_raise():
    for kw in [
        dict(metric="", threshold=0.5, alpha=0.05),
        dict(metric="r2", threshold=0.5, alpha=0.0),
        dict(metric="r2", threshold=0.5, alpha=1.0),
        dict(metric="r2", threshold="x", alpha=0.05),
    ]:
        try:
            prereg.canonical_spec(**kw)
            assert False, f"expected ValueError for {kw}"
        except ValueError:
            pass


TESTS = [
    test_canonical_spec_is_json_and_sorted,
    test_commit_determinism_same_spec_same_hash,
    test_commit_invariant_to_forbidden_field_order,
    test_commit_invariant_to_float_noise,
    test_relaxed_threshold_changes_hash,
    test_any_field_change_changes_hash,
    test_registry_missing_file_is_empty,
    test_registry_append_count_read,
    test_relaxed_recorded_alongside_original,
    test_registry_tolerates_corrupt_line,
    test_registry_atomic_no_tmp_left_behind,
    test_bad_inputs_raise,
]


def run(tests):
    p = f = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return p, f


def main():
    _, fails = run(TESTS)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
