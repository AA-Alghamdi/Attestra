"""LEAK-SAFE SPLITTERS -- group-aware, temporal (forward-chaining + embargo), and stratified.

WHY THIS EXISTS
---------------
The most common way a "great" certified result turns out to be a lie is a leaky split: the same patient
in train and test (the model memorizes the patient, not the disease); the same agent/customer in the
call-ender's train and sealed sets (it learns the agent, not the intent); a future row used to predict the
past. The frozen certifier is honest about the NUMBER it computes, but it cannot know your split leaked.
This module produces train/val/sealed index partitions that PROVABLY do not leak along the declared axis,
and exposes `assert_no_leakage` so the property is checked, not assumed.

This is the split-design counterpart to the frozen leak audit (vectorforge.science.audit). It changes no
certificate; it produces the partitions the rest of the pipeline certifies on.

CONTRACT: numpy only. Deterministic given a seed. No estimator, no certifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Fractions = Tuple[float, float, float]
DEFAULT_FRACTIONS: Fractions = (0.6, 0.2, 0.2)


class SplitError(ValueError):
    pass


@dataclass
class SplitResult:
    train_idx: List[int]
    val_idx: List[int]
    sealed_idx: List[int]
    method: str
    embargoed_idx: List[int]   # rows intentionally excluded from all splits (temporal embargo gap)

    def sizes(self) -> dict:
        return {"train": len(self.train_idx), "val": len(self.val_idx),
                "sealed": len(self.sealed_idx), "embargoed": len(self.embargoed_idx)}

    def all_disjoint(self) -> bool:
        s = [set(self.train_idx), set(self.val_idx), set(self.sealed_idx)]
        return (len(s[0] & s[1]) == 0 and len(s[0] & s[2]) == 0 and len(s[1] & s[2]) == 0)

    def assert_no_leakage(self, *, groups: Optional[Sequence] = None,
                          times: Optional[Sequence] = None) -> None:
        """Prove the declared no-leakage property. Raises SplitError on any violation."""
        if not self.all_disjoint():
            raise SplitError("splits overlap (a row appears in more than one of train/val/sealed)")
        if groups is not None:
            g = np.asarray(groups, dtype=object)
            gt, gv, gs = set(g[self.train_idx]), set(g[self.val_idx]), set(g[self.sealed_idx])
            if (gt & gs) or (gv & gs) or (gt & gv):
                bad = (gt & gs) | (gv & gs) | (gt & gv)
                raise SplitError(f"GROUP LEAKAGE: groups straddle splits: {sorted(map(str, bad))[:8]}")
        if times is not None:
            t = np.asarray(times, dtype=float)
            # forward-chaining requires max(train) <= min(val) and max(val) <= min(sealed) (ignoring embargo)
            if self.train_idx and self.val_idx and t[self.train_idx].max() > t[self.val_idx].min():
                raise SplitError("TEMPORAL LEAKAGE: a training row is later than a validation row")
            if self.val_idx and self.sealed_idx and t[self.val_idx].max() > t[self.sealed_idx].min():
                raise SplitError("TEMPORAL LEAKAGE: a validation row is later than a sealed row")


def _check_fractions(fr: Fractions) -> None:
    if len(fr) != 3 or any(f < 0 for f in fr) or abs(sum(fr) - 1.0) > 1e-6:
        raise SplitError(f"fractions must be three non-negative numbers summing to 1; got {fr}")


def stratified_split(labels: Sequence, fractions: Fractions = DEFAULT_FRACTIONS, *,
                     seed: int = 0) -> SplitResult:
    """Class-balanced split: each class's rows are partitioned by `fractions` independently, so train/val/
    sealed keep (approximately) the same class proportions. Deterministic given seed."""
    _check_fractions(fractions)
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val, sealed = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        train += idx[:n_tr].tolist()
        val += idx[n_tr:n_tr + n_va].tolist()
        sealed += idx[n_tr + n_va:].tolist()
    return SplitResult(sorted(train), sorted(val), sorted(sealed), "stratified", [])


def group_split(groups: Sequence, fractions: Fractions = DEFAULT_FRACTIONS, *,
                seed: int = 0) -> SplitResult:
    """Group-aware split: whole GROUPS are assigned to a single split, so a group never straddles splits
    (no patient/agent/customer leakage). Groups are shuffled then greedily packed to hit the target row
    fractions. Deterministic given seed."""
    _check_fractions(fractions)
    g = np.asarray(groups, dtype=object)
    uniq = list(dict.fromkeys(g.tolist()))           # stable unique order
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    counts = {u: int(np.sum(g == u)) for u in uniq}
    n = len(g)
    targets = [fractions[0] * n, fractions[1] * n, fractions[2] * n]
    buckets: List[List] = [[], [], []]
    filled = [0, 0, 0]
    for u in uniq:
        # assign group to the split most UNDER its target (largest remaining deficit), favoring sealed/val
        # only when they still have deficit so train doesn't starve them on tiny pools.
        deficits = [targets[i] - filled[i] for i in range(3)]
        j = int(np.argmax(deficits))
        buckets[j].append(u)
        filled[j] += counts[u]
    sel = lambda us: sorted(int(i) for i in np.where(np.isin(g, us))[0]) if us else []
    return SplitResult(sel(buckets[0]), sel(buckets[1]), sel(buckets[2]), "grouped", [])


def temporal_split(times: Sequence, fractions: Fractions = DEFAULT_FRACTIONS, *,
                   embargo: int = 0) -> SplitResult:
    """Forward-chaining split with an EMBARGO gap: sort rows by time, take the earliest `fractions[0]` as
    train, the next as val, the latest as sealed, and drop `embargo` rows at each boundary (excluded from
    all splits) so autocorrelation can't leak the future into the past. No randomness (time-ordered)."""
    _check_fractions(fractions)
    if embargo < 0:
        raise SplitError("embargo must be >= 0")
    t = np.asarray(times, dtype=float)
    order = np.argsort(t, kind="stable")
    n = len(order)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    train = order[:n_tr]
    # embargo gap after train
    va_start = min(n, n_tr + embargo)
    val = order[va_start:va_start + n_va]
    se_start = min(n, va_start + n_va + embargo)
    sealed = order[se_start:]
    used = set(train.tolist()) | set(val.tolist()) | set(sealed.tolist())
    embargoed = [int(i) for i in order.tolist() if i not in used]
    return SplitResult(sorted(int(i) for i in train), sorted(int(i) for i in val),
                       sorted(int(i) for i in sealed), "temporal", sorted(embargoed))


def make_leaksafe_splits(n: int, *, labels: Optional[Sequence] = None, groups: Optional[Sequence] = None,
                         times: Optional[Sequence] = None, embargo: int = 0,
                         fractions: Fractions = DEFAULT_FRACTIONS, seed: int = 0) -> SplitResult:
    """Pick the correct leak-safe splitter from what's declared: time_key -> temporal; group_key ->
    grouped; labels -> stratified; otherwise a plain shuffled split. The first applicable axis wins in the
    order temporal > grouped > stratified, because a temporal/group leak is the more dangerous one."""
    if times is not None:
        return temporal_split(times, fractions, embargo=embargo)
    if groups is not None:
        return group_split(groups, fractions, seed=seed)
    if labels is not None:
        return stratified_split(labels, fractions, seed=seed)
    _check_fractions(fractions)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    return SplitResult(sorted(idx[:n_tr].tolist()), sorted(idx[n_tr:n_tr + n_va].tolist()),
                       sorted(idx[n_tr + n_va:].tolist()), "shuffled", [])


__all__ = ["SplitResult", "SplitError", "stratified_split", "group_split", "temporal_split",
           "make_leaksafe_splits", "DEFAULT_FRACTIONS"]
