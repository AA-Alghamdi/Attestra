"""Hermetic locks for the certified data-ceiling curve (scripts/data_ceiling_curve.py).

No network, no cached embeddings: a synthetic arena with a KNOWN data dependence proves the curve machinery
is honest --
  * a genuinely data-limited task (XOR signal a strong head only resolves with enough labels) is CERTIFIED
    data-limited: the min->max slope is large, positive, and survives BH-FDR;
  * a pure-noise task (labels independent of features) is NOT certified -- the procedure cannot manufacture a
    data slope where none exists;
  * BH-FDR over the two tasks keeps only the signal;
  * the sweep never trains or selects on the sealed/val rows (select-then-bound), and is deterministic.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_ceiling_curve import _budget_grid, _subsample, certify_tasks, eval_task  # noqa: E402


def _split(y, seed=1):
    """Fixed, class-balanced sealed(30/class)/val(20/class)/train-pool(rest) split over the row ids."""
    idx = np.arange(len(y))
    rng = np.random.RandomState(seed)
    test, val, tr = [], [], []
    for c in (0, 1):
        ci = idx[y == c]
        rng.shuffle(ci)
        test += list(ci[:30])
        val += list(ci[30:50])
        tr += list(ci[50:])
    return np.array(sorted(tr)), np.array(sorted(val)), np.array(sorted(test))


def _signal(npc=220, seed=7, sigma=0.6, noise_dims=20):
    """An XOR over two noisy informative dims buried in `noise_dims` pure-noise dims: a strong tree head needs
    enough labels (samples in all four quadrants) to resolve it, so accuracy climbs with the training budget."""
    rng = np.random.RandomState(seed)
    n = 2 * npc
    a = rng.choice([-1.0, 1.0], size=n)
    b = rng.choice([-1.0, 1.0], size=n)
    y = ((a * b) > 0).astype(int)
    cols = [a + sigma * rng.randn(n), b + sigma * rng.randn(n)] + [rng.randn(n) for _ in range(noise_dims)]
    return np.stack(cols, axis=1).astype("float32"), y


def test_budget_grid_is_geometric_and_capped_by_the_pool():
    assert _budget_grid(56) == [6, 12, 24, 48, 56]          # full pool always included as the top of the curve
    assert _budget_grid(163) == [6, 12, 24, 48, 96, 163]
    assert _budget_grid(8) == [6, 8]
    assert _budget_grid(4) == [4]                           # floor below the smallest grid step


def test_subsample_is_balanced_deterministic_and_inside_the_train_pool():
    X, y = _signal()
    tr, val, test = _split(y)
    sub = _subsample(tr, y, 12, seed=0)
    assert sub.tolist() == _subsample(tr, y, 12, seed=0).tolist()        # deterministic
    assert np.bincount(y[sub]).tolist() == [12, 12]                      # balanced per class
    assert set(sub.tolist()).issubset(set(tr.tolist()))                  # never outside the train pool
    assert not (set(sub.tolist()) & set(test.tolist()))                  # never the sealed rows
    assert not (set(sub.tolist()) & set(val.tolist()))                   # never the val rows


def test_data_limited_signal_is_certified_and_pure_noise_is_not():
    X_sig, y = _signal()
    tr, val, test = _split(y)
    grid = _budget_grid(int(min(np.bincount(y[tr]))))

    sig = eval_task(X_sig, y, tr, val, test, grid, seeds=3)
    X_noise = np.random.RandomState(3).randn(len(y), 22).astype("float32")
    noise = eval_task(X_noise, y, tr, val, test, grid, seeds=3)

    # the signal curve climbs a lot from the smallest budget to the full pool; the noise curve does not move
    assert sig["slope_min_to_max"]["lift"] > 0.2
    assert sig["slope_min_to_max"]["p"] < 0.01
    assert abs(noise["slope_min_to_max"]["lift"]) < 0.1
    assert noise["slope_min_to_max"]["p"] > 0.1

    per_task = {"signal": sig, "noise": noise}
    survivors = certify_tasks(per_task)
    assert survivors == ["signal"]                                       # BH-FDR keeps only the real slope
    assert per_task["signal"]["data_limited_certified"] is True
    assert per_task["noise"]["data_limited_certified"] is False
    assert per_task["signal"]["verdict"].startswith("DATA-LIMITED")
    assert per_task["noise"]["verdict"].startswith("FLAT")


def test_curve_is_deterministic_across_runs():
    X, y = _signal()
    tr, val, test = _split(y)
    grid = _budget_grid(int(min(np.bincount(y[tr]))))
    a = eval_task(X, y, tr, val, test, grid, seeds=2)
    b = eval_task(X, y, tr, val, test, grid, seeds=2)
    assert a["slope_min_to_max"] == b["slope_min_to_max"]
    assert [c["acc_mean"] for c in a["curve"]] == [c["acc_mean"] for c in b["curve"]]
