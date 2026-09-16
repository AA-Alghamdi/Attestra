"""Vision modality tests (NEW module vfplatform/vision.py).

Two layers:
  1. UNIT -- the ImageFeaturizer mirrors the TabularFeaturizer interface EXACTLY (fit(train, all) ->
     transform(rows) -> a fixed-width float matrix), normalizes pixels to [0,1], appends a 2x2 average-pool
     block for square images, and is deterministic. The harness reuses the EXISTING classification catalog.
  2. END-TO-END -- run_goal_loop on vision_demo() (sklearn digits as 8x8 image records) drives the EXISTING
     loop: select on validation -> ONE counted sealed peek through the EXISTING frozen certifier. We assert
     the decision is one of {certified, honest_stop, best_effort, do_not_certify} and, when a certificate is
     produced, that it carries the SAME frozen-certifier fields (no new promotion path was introduced).

Read-only on every existing module: vision imports Harness/catalog from harness.py and run_goal_loop from
loop.py without modifying them. CPU-only, no network, no torch.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_vision.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vfplatform.vision import VisionClassificationHarness, ImageFeaturizer, vision_demo
from vfplatform.harness import catalog_for, TabularFeaturizer
from vfplatform.loop import run_goal_loop


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


def _subsample(rows, per_class=18, seed=0):
    """A small, class-balanced subsample so the e2e loop is fast yet has every digit class represented."""
    rng = np.random.RandomState(seed)
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(r)
    out = []
    for lab, group in sorted(by.items()):
        idx = rng.permutation(len(group))[:per_class]
        out += [group[i] for i in idx]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- unit: featurizer
def test_featurizer_interface_matches_tabular():
    """Same .fit(rows_for_schema, all_rows)->self and .transform(rows)->ndarray contract as the tabular
    featurizer, so the loop is unchanged."""
    rows = vision_demo()["records"][:40]
    feat = ImageFeaturizer()
    assert feat.fit(rows[:20], rows) is feat, "fit must return self (like TabularFeaturizer)"
    X = feat.transform(rows[:5])
    assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == 5
    # interface parity check against the real TabularFeaturizer signature
    tf = TabularFeaturizer()
    assert hasattr(tf, "fit") and hasattr(tf, "transform")


def test_featurizer_shape_and_pooling():
    rows = vision_demo()["records"][:60]
    feat = ImageFeaturizer().fit(rows[:30], rows)
    X = feat.transform(rows[:10])
    # 64 raw pixels (8x8) + 16 pooled (4x4 from a 2x2 average-pool) = 80 features
    assert X.shape == (10, 80), X.shape


def test_featurizer_normalizes_to_unit_range():
    rows = vision_demo()["records"][:60]
    feat = ImageFeaturizer().fit(rows[:30], rows)
    X = feat.transform(rows[:30])
    assert X.min() >= -1e-9 and X.max() <= 1.0 + 1e-9, (X.min(), X.max())


def test_featurizer_deterministic():
    rows = vision_demo()["records"][:50]
    feat = ImageFeaturizer().fit(rows[:25], rows)
    assert np.allclose(feat.transform(rows[:20]), feat.transform(rows[:20]))


def test_featurizer_handles_nested_2d_and_short_rows():
    img2d = [{"features": [[float((i + a + b) % 16) for b in range(8)] for a in range(8)],
              "target": str(i % 2)} for i in range(12)]
    feat = ImageFeaturizer().fit(img2d, img2d)
    X = feat.transform(img2d)
    assert X.shape == (12, 80)
    # a short/malformed row is padded, not crash-causing: width stays constant
    short = [{"features": [1.0, 2.0, 3.0], "target": "0"}]
    Xs = feat.transform(short)
    assert Xs.shape == (1, 80)


def test_harness_reuses_classification_catalog():
    """The vision harness draws from the EXISTING tabular classification zoo -- no new model family."""
    h = VisionClassificationHarness(task_type="multiclass")
    assert set(h.catalog().keys()) == set(catalog_for("tabular", "multiclass").keys())
    # modality identity is "vision"; the FROZEN-pipeline identity is "tabular" (pixels flow as the legitimate
    # numeric features the frozen leakage auditor + certifier already validate -- see the harness docstring).
    assert h.modality == "vision" and h.task_type == "multiclass" and h.default_metric == "accuracy"
    assert h.kind == "tabular", "the frozen-pipeline kind must be tabular so pixels flow through the audit"
    moves = h.moves(seeds=(0,))
    assert len(moves) >= 2 and any(m.name == "baseline" for m in moves)


# --------------------------------------------------------------------------- e2e: the existing certifier
def test_end_to_end_vision_uses_existing_certifier():
    demo = vision_demo()
    rows = _subsample(demo["records"], per_class=18)         # ~180 rows, all 10 classes
    h = VisionClassificationHarness(task_type="multiclass")
    # the loop reads harness.kind (="tabular") for the FROZEN audit + catalog; passing kind="tabular" here is
    # the same pipeline identity (the kwarg is ignored when a harness is supplied, but we keep them in parity).
    res = run_goal_loop(
        rows, "classify these 8x8 handwritten digit images", harness=h,
        kind="tabular", task_type="multiclass", target_key="target", labels=demo["labels"],
        threshold=0.70, metric="accuracy", seeds=(0,), min_test_n=10, max_rounds=3,
        llm_propose=False, llm_enabled=False)

    # blocked is NOT acceptable: image pixels must flow through the frozen leakage audit as legitimate
    # tabular features (not be rejected as "non-tabular features"), then certify or honest-stop honestly.
    assert res.decision != "blocked", "vision pixels must pass the frozen tabular audit, not be blocked"
    assert res.decision in ("certified", "honest_stop", "best_effort", "do_not_certify"), res.decision
    assert res.provider == "local-cpu", res.provider     # the in-process CPU provider (no GPU/worker)
    # If a certificate was produced, it must carry the FROZEN certifier's fields (it came through
    # certify_on_sealed -> science) -- NOT a new vision-specific promotion path.
    if res.certificate is not None:
        cert = res.certificate
        for key in ("observed", "lower_bound", "theta", "certified", "checks", "peeks"):
            assert key in cert, f"certificate missing frozen-certifier field {key!r}: {sorted(cert)}"
        assert cert["checks"] == 1, "the sealed peek is the existing one-counted peek (checks=1)"
        assert cert["winner_family"] in h.catalog() or "|" in cert["winner_family"], (
            "the certified winner is a family from the EXISTING classification catalog")
        # certified iff the lower bound genuinely clears theta (the frozen invariant; not a vision shortcut)
        if cert["certified"]:
            assert cert["lower_bound"] > cert["theta"], cert
    else:
        # an honest stop spends NO peek -> no certificate; that is the only no-certificate decision allowed
        assert res.decision in ("honest_stop",), res.decision
    print(f"    e2e decision={res.decision} "
          + ("" if res.certificate is None
             else f"observed={res.certificate['observed']} lb={res.certificate['lower_bound']} "
                  f"n={res.certificate['n']}"))


def test_no_new_cert_path_introduced():
    """The vision module must not define its own certifier / sealed-peek / threshold logic. It reuses the
    frozen path entirely. We assert it exposes no such symbol."""
    import vfplatform.vision as v
    for forbidden in ("certify", "certify_on_sealed", "SealedTest", "certify_accuracy", "lower_bound",
                      "_val_lower_bound"):
        assert not hasattr(v, forbidden), f"vision must not define its own promotion path: {forbidden}"


TESTS = [test_featurizer_interface_matches_tabular, test_featurizer_shape_and_pooling,
         test_featurizer_normalizes_to_unit_range, test_featurizer_deterministic,
         test_featurizer_handles_nested_2d_and_short_rows, test_harness_reuses_classification_catalog,
         test_end_to_end_vision_uses_existing_certifier, test_no_new_cert_path_introduced]


def main():
    print("== test_vision (image classification via the FROZEN certifier) ==")
    p, f = run(TESTS)
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
