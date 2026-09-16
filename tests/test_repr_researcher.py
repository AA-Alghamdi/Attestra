"""HERMETIC tests for the autonomous representation researcher's HONESTY-critical invariant: a challenger
is promoted to champion if and ONLY if the frozen Tier-3 sealed certifier certifies it, and that decision
is a pure function of the SEALED correctness vectors -- validation accuracy can only screen/order, never
promote. No network, no pretrained weights, no FGVC download: a synthetic FakeArena supplies fully-controlled
correctness vectors, while the suite-level statistics are the REAL frozen functions (mcnemar_pvalue,
benjamini_hochberg, science.clopper_pearson_lower) -- so the actual certifier path decides, byte-for-byte.

The three properties locked here:
  (1) PROMOTE-ON-FDR-SURVIVAL: a challenger that is genuinely better on the SEALED rows (discordant pairs
      favour it) is promoted; the frozen FDR survival is what mints the promotion.
  (2) NO VALIDATION LURE: a challenger with the HIGHEST validation accuracy but NO sealed-rows advantage is
      selected first (best-val-first) yet is REJECTED -- validation never promotes.
  (3) FLIPPING SEALED LABELS BLOCKS PROMOTION: take the genuine winner, flip ONLY its sealed correctness
      (keep its high validation), and it is no longer promoted -- proving the decision rides on the sealed
      test alone and sealed labels never leak in through the validation/selection path.
"""
import hashlib
import os

from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue
from vfplatform.repr_researcher import Arena, Encoder, ReprResearcher, TaskMeasure
from vectorforge.science import clopper_pearson_lower

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}

# ----------------------------------------------------------------------------------------------------------
# Synthetic per-example correctness, ALIGNED across challenger/champion so McNemar discordant pairs are exact.
# Each task has n=60 sealed rows. The champion ("weak") is correct on rows [0..29] only (acc 0.50).
N = 60
TASKS = ["t0", "t1", "t2"]

def _vec(correct_through):
    """Correctness vector of length N: rows [0..correct_through) are 1 (correct), the rest 0."""
    return [1] * correct_through + [0] * (N - correct_through)

WEAK = _vec(30)        # champion: acc 0.50, correct on rows 0..29
STRONG = _vec(55)      # genuine winner: acc 0.917, correct on 0..54 -> +25 discordant wins, 0 losses vs WEAK
LURE_SEALED = _vec(30) # SAME sealed rows as champion (0 sealed advantage) ...
LURE_VAL = _vec(57)    # ... but the HIGHEST validation accuracy (0.95) -> selected first, must still lose
FUSE_VAL = _vec(36)    # fusion validation 0.60: clears the floor so the certify path is exercised ...
FUSE_SEALED = _vec(30) # ... but sealed acc 0.50 == the weak champion -> 0 lift, so it never certifies

# GOLD (never-peeked) correctness, INDEPENDENT of the sealed rows: the strong rep also wins on the gold set
# (so a genuine champion is gold-CONFIRMED), while the weak baseline sits at chance there.
GOLD_WEAK = _vec(30)    # baseline gold: acc 0.50
GOLD_STRONG = _vec(55)  # champion gold: acc 0.917 -> +25 discordant wins over the baseline on gold


class FakeArena(Arena):
    """A fully-controlled arena: measure() returns the stored correctness vectors; the suite statistics are
    the REAL frozen functions so the genuine certifier path makes every promotion decision."""

    tasks = list(TASKS)

    def __init__(self, *, flip_strong_sealed=False, flip_strong_gold=False, has_gold=True):
        self._flip = flip_strong_sealed
        # (val_correct, sealed_correct) per encoder tag.
        self._cfg = {
            "weak":   (WEAK, WEAK),
            "strong": (STRONG, [1 - x for x in STRONG] if flip_strong_sealed else STRONG),
            "lure":   (LURE_VAL, LURE_SEALED),
        }
        # gold correctness per tag (a DISJOINT never-peeked set). flip_strong_gold makes the sealed-promoted
        # champion LOSE on gold -> the gold confirmation must then fail, proving gold never drives promotion.
        self._has_gold = bool(has_gold)
        self._gold = {
            "weak":   GOLD_WEAK,
            "strong": [1 - x for x in GOLD_STRONG] if flip_strong_gold else GOLD_STRONG,
            "lure":   GOLD_WEAK,
        }
        self.gold_calls = []   # records the order/identity of gold reads (must be post-hoc: champion, baseline)

    def measure(self, encoder_tag):
        val, sealed = self._cfg[encoder_tag]
        acc = sum(sealed) / len(sealed)
        return {t: TaskMeasure(sealed_correct=list(sealed), val_correct=list(val), acc=acc) for t in TASKS}

    def gold_measure(self, name):
        if not self._has_gold or name not in self._gold:
            return None
        self.gold_calls.append(name)
        return {t: list(self._gold[name]) for t in TASKS}

    def fuse_measure(self, tag_a, tag_b):
        acc = sum(FUSE_SEALED) / len(FUSE_SEALED)
        return {t: TaskMeasure(sealed_correct=list(FUSE_SEALED), val_correct=list(FUSE_VAL), acc=acc)
                for t in TASKS}

    def mcnemar(self, chal_correct, base_correct):
        return mcnemar_pvalue(list(chal_correct), list(base_correct))

    def bh(self, pvalues, alpha):
        return list(benjamini_hochberg(list(pvalues), alpha=alpha))

    def lower_bound(self, correct):
        k, n = int(sum(correct)), len(correct)
        return round(clopper_pearson_lower(k, n, 0.1), 4)


REGISTRY = [
    Encoder("weak",   "weak champion",   "w", 0, 10.0),
    Encoder("strong", "strong rep",      "s", 0, 50.0),
    Encoder("lure",   "validation lure", "l", 0, 50.0),
]


def _researcher(arena, registry=REGISTRY):
    return ReprResearcher(registry, arena, start_tag="weak", alpha=0.1, theta_floor=0.5,
                          peek_budget=12, competence_ceiling=0.90)


def test_frozen_certifier_unchanged():
    """Guard: the test certifies through the frozen files; they must be byte-identical to the pinned hashes."""
    got = {}
    for rel in FROZEN_EXPECTED:
        with open(os.path.join(_ROOT, rel), "rb") as fh:
            got[rel] = hashlib.sha256(fh.read()).hexdigest()[:8]
    assert got == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {got} != {FROZEN_EXPECTED}"


def test_promotes_only_on_fdr_survival_and_rejects_validation_lure():
    """The genuine sealed winner ('strong') is promoted to champion; the higher-validation 'lure' (no sealed
    advantage) is selected first yet REJECTED. Promotion rides on frozen FDR survival, never on validation."""
    cert = _researcher(FakeArena()).run()

    assert cert.champion == "strong", cert.champion
    promoted = [p.to_tag for p in cert.promotions]
    assert "strong" in promoted
    assert "lure" not in promoted                      # the validation lure never becomes champion

    rejected = {r.tag: r for r in cert.rejections}
    assert "lure" in rejected
    assert rejected["lure"].survivors == []            # zero sealed FDR survivors despite top validation acc

    # the promotion of 'strong' must carry real frozen FDR survivors (>0) with positive lift
    strong_promo = next(p for p in cert.promotions if p.to_tag == "strong")
    assert len(strong_promo.survivors) > 0
    assert strong_promo.mean_lift > 0


def test_flipping_sealed_labels_blocks_promotion():
    """Flip ONLY the genuine winner's SEALED correctness (keep its high validation). It now passes the
    validation competence screen and is selected, but the frozen Tier-3 cannot certify it -> NO promotion.
    The champion stays 'weak'. This proves the decision depends on sealed rows alone; validation cannot leak
    a promotion through."""
    registry = [REGISTRY[0], REGISTRY[1]]              # weak + strong only (drop the lure)
    cert = _researcher(FakeArena(flip_strong_sealed=True), registry=registry).run()

    assert cert.champion == "weak", cert.champion       # flipped sealed -> no climb at all
    assert [p.to_tag for p in cert.promotions] == []
    rejected = {r.tag: r for r in cert.rejections}
    assert "strong" in rejected                         # selected (high val) but not certified on sealed rows
    assert rejected["strong"].survivors == []


def test_promotion_is_invariant_to_validation_above_floor():
    """Lowering the winner's VALIDATION accuracy (while keeping it above the competence floor and its sealed
    rows unchanged) must NOT change the promotion -- validation only screens/orders, it never decides."""
    arena = FakeArena()
    arena._cfg["strong"] = (_vec(42), STRONG)           # val lowered to 0.70 (lb clears the 0.50 floor);
    cert = _researcher(arena).run()                     # sealed rows unchanged
    assert cert.champion == "strong"                    # still promoted -> sealed rows drive the decision


def test_gold_confirmation_is_post_hoc_and_confirms_a_genuine_champion():
    """The gold set is read EXACTLY ONCE per principal (the final champion, then the start baseline) and only
    AFTER the climb -- never inside the promotion loop. A genuine champion that wins on the disjoint gold set
    is CONFIRMED, with real frozen FDR survivors and the champion's gold lower bound above the baseline's."""
    arena = FakeArena()
    cert = _researcher(arena).run()
    assert cert.champion == "strong"
    # gold was queried post-hoc, exactly once each, champion first then baseline -- not per round.
    assert arena.gold_calls == ["strong", "weak"], arena.gold_calls
    gc = cert.gold_confirmation
    assert gc is not None
    assert gc["champion"] == "strong" and gc["baseline"] == "weak"
    assert gc["confirmed"] is True
    assert len(gc["survivors"]) > 0 and gc["mean_lift"] > 0
    assert gc["champion_gold_lb"] > gc["baseline_gold_lb"]


def test_gold_confirmation_can_fail_even_when_sealed_promotes():
    """Gold NEVER drives promotion: flip ONLY the champion's GOLD correctness (its sealed rows still win, so it
    is still promoted to champion), and the gold confirmation must come back NOT confirmed. This proves gold
    is an independent, never-peeked check that can contradict the sealed climb -- it cannot launder a win."""
    arena = FakeArena(flip_strong_gold=True)
    cert = _researcher(arena).run()
    assert cert.champion == "strong"                    # sealed rows unchanged -> still promoted
    gc = cert.gold_confirmation
    assert gc is not None
    assert gc["confirmed"] is False                     # but the never-peeked gold set does NOT confirm it
    assert gc["survivors"] == []


def test_gold_confirmation_absent_when_arena_supplies_no_gold():
    """A data-limited domain with no disjoint gold partition emits NO gold confirmation (None), rather than
    fabricating one -- the honest degradation."""
    arena = FakeArena(has_gold=False)
    cert = _researcher(arena).run()
    assert cert.champion == "strong"
    assert cert.gold_confirmation is None
    assert arena.gold_calls == []


def test_session_multiplicity_discloses_looks_and_is_robust_for_a_genuine_champion():
    """The certificate always carries an honest session-multiplicity report: the count of sealed
    certifications spent, the family-wise Bonferroni threshold alpha/M that implies, and a conservative
    re-test of the FINAL champion vs the START baseline at that threshold. For a genuine champion the win is
    so large it survives the Bonferroni correction over every look (robust), and matches the independent
    gold confirmation."""
    cert = _researcher(FakeArena()).run()
    mp = cert.multiplicity
    assert mp is not None
    m = mp["sealed_comparisons"]
    assert m == cert.peeks_used and m > 1                      # >1 look was spent climbing
    assert mp["mcnemar_tests_total"] == m * len(TASKS)         # M suite-looks x n_tasks paired tests
    assert mp["per_comparison_fdr_alpha"] == 0.1
    assert abs(mp["session_bonferroni_alpha"] - 0.1 / m) < 1e-9
    assert mp["baseline"] == "weak" and mp["champion"] == "strong"
    # the +0.42/task win has McNemar p ~ 0.5^25, far below alpha/M, so it survives the family-wise correction
    assert set(mp["bonferroni_survivors_session"]) == set(TASKS)
    assert set(mp["fdr_survivors_nominal"]) == set(TASKS)
    assert mp["robust_to_session_multiplicity"] is True
    assert mp["gold_independent_confirmation"] is True         # agrees with the never-peeked gold read


def test_session_multiplicity_is_honest_when_no_climb_happens():
    """When nothing certifies (flipped sealed -> champion stays the start baseline), the report says so
    honestly: champion == start, zero per-task lift, and the Bonferroni re-test yields NO survivors (the
    selection-bias correction cannot manufacture a win that the sealed rows do not contain)."""
    registry = [REGISTRY[0], REGISTRY[1]]
    cert = _researcher(FakeArena(flip_strong_sealed=True), registry=registry).run()
    assert cert.champion == "weak"
    mp = cert.multiplicity
    assert mp is not None
    assert mp["champion"] == "weak" and mp["baseline"] == "weak"
    assert mp["sealed_comparisons"] >= 1                       # looks WERE spent (and all failed)
    assert all(pt["lift"] == 0.0 for pt in mp["champion_vs_start_sealed"])
    assert mp["bonferroni_survivors_session"] == []
    assert mp["fdr_survivors_nominal"] == []
    assert mp["robust_to_session_multiplicity"] is False
