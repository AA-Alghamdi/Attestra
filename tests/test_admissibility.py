"""Deterministic admissibility-inspector tests (NEW module vfplatform/admissibility.py).

Proves the inspector is an HONEST structural gate -- NOT a certifier. It detects the task type from the
data alone and flags the hard issues that make a run inadmissible (constant target, a feature identical to
the target, too few rows), plus the soft duplicate-heavy warning. Read-only: nothing here touches the
frozen certifier, the sealed peek, or any existing module.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_admissibility.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import admissibility


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


def test_detects_binary_and_metric():
    rows = [{"features": {"a": float(i), "b": float(i % 7)}, "target": str(i % 2)} for i in range(50)]
    r = admissibility.inspect(rows)
    assert r["admissible"] is True, r["verdict"]
    assert r["suggested_task_type"] == "binary", r["suggested_task_type"]
    assert r["suggested_metric"] == "accuracy"
    assert r["n_rows"] == 50 and r["n_features"] == 2
    assert r["suggested_target"] == "target"


def test_detects_multiclass():
    rows = [{"features": {"a": float(i)}, "target": str(i % 4)} for i in range(60)]
    r = admissibility.inspect(rows)
    assert r["admissible"] is True
    assert r["suggested_task_type"] == "multiclass"
    assert r["suggested_metric"] == "f1_macro"


def test_detects_regression():
    rows = [{"features": {"a": float(i)}, "target": float(i) * 0.7} for i in range(40)]
    r = admissibility.inspect(rows)
    assert r["suggested_task_type"] == "regression"
    assert r["suggested_metric"] == "r2"


def test_flags_constant_target():
    rows = [{"features": {"a": float(i)}, "target": "only_one"} for i in range(40)]
    r = admissibility.inspect(rows)
    assert r["admissible"] is False, "a constant target leaves nothing to learn -> inadmissible"
    assert "constant target" in r["verdict"]
    assert any("constant target" in i for i in r["issues"])


def test_flags_feature_equals_target_leak():
    rows = [{"features": {"a": float(i), "leak": str(i % 3)}, "target": str(i % 3)} for i in range(45)]
    r = admissibility.inspect(rows)
    assert r["admissible"] is False, "a feature identical to the target is trivial leakage -> inadmissible"
    assert "leakage" in r["verdict"]
    assert any("leak" in i for i in r["issues"])


def test_flags_too_few_rows():
    rows = [{"features": {"a": float(i)}, "target": str(i % 2)} for i in range(6)]
    r = admissibility.inspect(rows)
    assert r["admissible"] is False, f"{len(rows)} rows is below the documented floor"
    assert "too few rows" in r["verdict"]


def test_flat_record_shape():
    rows = [{"a": float(i), "b": float(i % 5), "target": str(i % 2)} for i in range(40)]
    r = admissibility.inspect(rows)
    assert r["admissible"] is True
    assert r["n_features"] == 2, "flat (un-nested) records expose every non-target key as a feature"


def test_explicit_target_key():
    rows = [{"features": {"a": float(i)}, "label": str(i % 2)} for i in range(40)]
    r = admissibility.inspect(rows, target_key="label")
    assert r["suggested_target"] == "label"
    assert r["admissible"] is True


def test_duplicate_heavy_is_soft_warning():
    base = {"features": {"a": 1.0, "b": 2.0}, "target": "x"}
    other = [{"features": {"a": float(i), "b": float(i)}, "target": "y"} for i in range(5)]
    rows = [dict(base) for _ in range(35)] + other     # >50% exact dups, but 2 classes present
    r = admissibility.inspect(rows)
    assert r["admissible"] is True, "duplicate-heavy is RISKY, not hard-fatal"
    assert any("duplicate-heavy" in i for i in r["issues"]), r["issues"]


def test_empty_and_nonrectangular_decline():
    assert admissibility.inspect([])["admissible"] is False
    ragged = [{"features": {"a": 1.0}, "target": "x"}, {"features": {"b": 2.0}, "target": "y"}] * 20
    r = admissibility.inspect(ragged)
    assert r["admissible"] is False
    assert "non-rectangular" in r["verdict"]


def test_schema_profile():
    rows = [{"features": {"num": float(i), "cat": ("p" if i % 2 else "q"), "miss": None},
             "target": str(i % 2)} for i in range(30)]
    r = admissibility.inspect(rows)
    sc = r["schema"]
    assert sc["num"]["numeric"] is True
    assert sc["cat"]["numeric"] is False and sc["cat"]["n_unique"] == 2
    assert sc["miss"]["n_missing"] == 30
    assert any("entirely missing" in i for i in r["issues"])


def test_determinism():
    rows = [{"features": {"a": float(i)}, "target": str(i % 3)} for i in range(40)]
    assert admissibility.inspect(rows) == admissibility.inspect(rows)


def test_verdict_is_not_a_certificate():
    """An admissible verdict carries NO certificate keys -- it is an inspector decision, not a promotion."""
    rows = [{"features": {"a": float(i)}, "target": str(i % 2)} for i in range(40)]
    r = admissibility.inspect(rows)
    for forbidden in ("certified", "lower_bound", "observed", "peeks", "certificate"):
        assert forbidden not in r, f"admissibility must not emit a certificate field: {forbidden}"


TESTS = [test_detects_binary_and_metric, test_detects_multiclass, test_detects_regression,
         test_flags_constant_target, test_flags_feature_equals_target_leak, test_flags_too_few_rows,
         test_flat_record_shape, test_explicit_target_key, test_duplicate_heavy_is_soft_warning,
         test_empty_and_nonrectangular_decline, test_schema_profile, test_determinism,
         test_verdict_is_not_a_certificate]


def main():
    print("== test_admissibility (deterministic inspector gate) ==")
    p, f = run(TESTS)
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
