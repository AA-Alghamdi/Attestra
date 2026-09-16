"""Tests for vfplatform.run_context: bounded summarized history, overflow-raises
(no silent truncation), and a compact run-state that preserves the key signal."""

import pytest

from vfplatform.run_context import (
    RunContext,
    RoundSummary,
    CompactionError,
    chars_to_tokens,
    _compact_diagnosis,
)


def _diag(val_lb=0.6, val=0.7, headroom=0.1, ece=0.05, recall=0.55, below=True, extra=None):
    d = {
        "val": val,
        "val_lb": val_lb,
        "headroom": headroom,
        "ece": ece,
        "below_bar": below,
        "min_class_recall": recall,
    }
    if extra:
        d.update(extra)
    return d


# --------------------------------------------------------------------------
# (a) history stays bounded
# --------------------------------------------------------------------------

def test_detailed_tail_is_bounded_regardless_of_run_length():
    rc = RunContext(max_detailed_rounds=4, total_budget=10_000, per_entry_ceiling=1000)
    for i in range(50):
        rc.record_round(_diag(val_lb=0.5 + i * 0.001), [f"m{i}"])
    st = rc.run_state()
    # never more than max_detailed_rounds full summaries are kept
    assert len(st["recent_rounds"]) <= 4
    assert st["n_rounds"] == 50
    # the rest folded into the O(1) aggregate
    assert st["aggregate"]["rounds_folded"] == 50 - len(st["recent_rounds"])


def test_run_state_size_stays_within_budget_over_a_long_run():
    rc = RunContext(total_budget=2000, per_entry_ceiling=600, max_detailed_rounds=8, unit="tokens")
    for i in range(200):
        rc.record_round(_diag(val_lb=0.5), [f"stronger_model_{i % 3}", "regularize"])
    st = rc.run_state()
    assert rc._size(st) <= rc.total_budget
    # size must NOT grow with run length: assemble again after more rounds
    for i in range(200):
        rc.record_round(_diag(val_lb=0.5), [f"deep_model_{i % 4}"])
    st2 = rc.run_state()
    assert rc._size(st2) <= rc.total_budget


def test_moves_ever_tried_survives_aging_out_of_detailed_tail():
    rc = RunContext(max_detailed_rounds=2)
    rc.record_round(_diag(), ["alpha"])
    rc.record_round(_diag(), ["beta"])
    rc.record_round(_diag(), ["gamma"])
    rc.record_round(_diag(), ["delta"])
    # alpha/beta have aged out of the detailed tail but must still be remembered
    ever = rc.moves_ever_tried()
    assert set(ever) == {"alpha", "beta", "gamma", "delta"}
    assert set(rc.run_state()["moves_ever_tried"]) == {"alpha", "beta", "gamma", "delta"}


# --------------------------------------------------------------------------
# (b) overflow RAISES, does not silently truncate
# --------------------------------------------------------------------------

def test_per_entry_overflow_raises_not_truncates():
    rc = RunContext(per_entry_ceiling=5, unit="tokens")
    with pytest.raises(CompactionError) as ei:
        rc.record_round(_diag(), ["a_move_name_long_enough_to_overflow_the_tiny_ceiling"])
    err = ei.value
    assert err.size > err.budget
    assert err.unit == "tokens"
    assert err.field is not None
    # learn-from-the-incident: the error is structured/auditable
    assert err.as_dict()["error"] == "CompactionError"
    # and NOTHING was silently stored despite the failure
    assert rc.n_rounds == 0


def test_run_state_overflow_raises_when_even_minimal_state_too_big():
    # Tiny total budget that cannot hold even the minimal state (with no tail).
    rc = RunContext(total_budget=3, per_entry_ceiling=10_000, unit="tokens")
    rc.record_round(_diag(), ["stronger_model", "regularize", "calibrate", "deep_model"])
    with pytest.raises(CompactionError) as ei:
        rc.run_state()
    assert ei.value.size > ei.value.budget


def test_run_state_shrinks_tail_before_raising():
    # Budget big enough for the minimal state but not the full tail: it must
    # drop tail detail (oldest first) rather than raise -- the key signal lives
    # in moves_ever_tried/aggregate, so this is a legitimate compaction.
    rc = RunContext(total_budget=10_000, per_entry_ceiling=10_000, max_detailed_rounds=20)
    for i in range(20):
        rc.record_round(_diag(val_lb=0.5 + i * 0.001), [f"m{i}", "regularize"])
    full = rc.run_state()
    n_full_tail = len(full["recent_rounds"])
    # now ask for the same state under a tight budget
    rc.total_budget = 700
    tight = rc.run_state()
    assert len(tight["recent_rounds"]) < n_full_tail
    # the union of tried moves is unchanged despite the shrink
    assert set(tight["moves_ever_tried"]) == set(full["moves_ever_tried"])


# --------------------------------------------------------------------------
# (c) summary preserves the key signal (best val_lb + what was tried)
# --------------------------------------------------------------------------

def test_summary_preserves_best_val_lb():
    rc = RunContext()
    rc.record_round(_diag(val_lb=0.50), ["a"])
    rc.record_round(_diag(val_lb=0.62), ["b"])   # best
    rc.record_round(_diag(val_lb=0.58), ["c"])   # regression -> best stays 0.62
    st = rc.run_state()
    assert st["best_val_lb"] == 0.62
    assert st["best_val_lb_round"] == 1
    assert rc.best_val_lb == 0.62


def test_summary_preserves_what_was_tried():
    rc = RunContext()
    rc.record_round(_diag(), ["stronger_model"])
    rc.record_round(_diag(), ["regularize", "calibrate"])
    st = rc.run_state()
    assert set(st["moves_ever_tried"]) == {"stronger_model", "regularize", "calibrate"}


def test_plateau_signal_detects_no_improvement():
    rc = RunContext(max_detailed_rounds=4)
    rc.record_round(_diag(val_lb=0.50), ["a"])
    for _ in range(6):
        rc.record_round(_diag(val_lb=0.50), ["b"])  # flat -> plateau
    st = rc.run_state()
    assert st["rounds_since_improvement"] == 6
    assert st["plateaued"] is True


def test_plateau_resets_on_improvement():
    rc = RunContext()
    rc.record_round(_diag(val_lb=0.50), ["a"])
    rc.record_round(_diag(val_lb=0.50), ["b"])
    rc.record_round(_diag(val_lb=0.50), ["c"])
    assert rc.rounds_since_improvement == 2
    rc.record_round(_diag(val_lb=0.70), ["d"])  # improvement
    assert rc.rounds_since_improvement == 0


def test_diagnosis_is_compacted_to_known_keys():
    rc = RunContext()
    rc.record_round(_diag(extra={"junk_field": "x" * 500, "another": [1, 2, 3]}), ["a"])
    st = rc.run_state()
    latest = st["latest_diagnosis"]
    assert "junk_field" not in latest
    assert "another" not in latest
    assert set(latest.keys()) <= {"val", "val_lb", "headroom", "ece", "below_bar", "min_class_recall"}
    assert latest["val_lb"] == 0.6


def test_compact_diagnosis_handles_missing_and_none():
    # robust to partial diagnosis dicts (e.g. regression path has no min_class_recall)
    out = _compact_diagnosis({"val_lb": 0.5, "below_bar": True})
    assert out == {"val_lb": 0.5, "below_bar": True}
    assert _compact_diagnosis(None) == {}


def test_proposer_view_has_tried_and_diagnosis():
    rc = RunContext()
    rc.record_round(_diag(val_lb=0.55), ["stronger_model"])
    rc.record_round(_diag(val_lb=0.60), ["regularize"])
    view = rc.proposer_view(recent_rounds=2)
    assert set(view["tried"]) == {"stronger_model", "regularize"}
    assert view["best_val_lb"] == 0.60
    assert view["diagnosis"]["val_lb"] == 0.6
    assert "plateaued" in view


def test_val_lb_falls_back_to_diagnosis_when_not_passed():
    rc = RunContext()
    rc.record_round(_diag(val_lb=0.42), ["a"])  # val_lb only in the diagnosis dict
    assert rc.best_val_lb == 0.42


# --------------------------------------------------------------------------
# misc / invariants
# --------------------------------------------------------------------------

def test_moves_are_deduped_and_sorted_per_round():
    rc = RunContext()
    rs = rc.record_round(_diag(), ["b", "a", "b", "a"])
    assert rs.moves_tried == ["a", "b"]
    assert isinstance(rs, RoundSummary)


def test_chars_to_tokens_ceils_nonempty():
    assert chars_to_tokens(0) == 0
    assert chars_to_tokens(1) == 1
    assert chars_to_tokens(4) == 1
    assert chars_to_tokens(5) == 2


def test_constructor_validates_args():
    with pytest.raises(ValueError):
        RunContext(unit="bogus")
    with pytest.raises(ValueError):
        RunContext(total_budget=0)
    with pytest.raises(ValueError):
        RunContext(max_detailed_rounds=0)


def test_chars_unit_budget():
    rc = RunContext(total_budget=100_000, per_entry_ceiling=100_000, unit="chars")
    rc.record_round(_diag(), ["a"])
    st = rc.run_state()
    assert rc._size(st) <= rc.total_budget
