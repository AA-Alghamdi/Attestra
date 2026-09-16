"""PROPOSE-side literature / method KNOWLEDGE SURFACE (advisory, NON-BINDING).

This is a read-advice node the PROPOSE step may consult BEFORE it builds its bounded grid: given a
TRUSTED task profile (shapes, not rows) and the round's measured DIAGNOSIS, it returns structured
method/feature SUGGESTIONS -- "given high-dim + nonlinear headroom, try mutual-info selection +
gradient boosting" -- expressed as (a) catalog FAMILY HINTS (names the proposer already knows) and
(b) FEATURE MOVES (named, parameter-light transforms the proposer can choose to apply). It also
emits a free-text rationale and, when available, a literature/dataset NOTE.

WHY THIS IS SAFE (the integrity contract):
  * It is ADVISORY ONLY. It never builds an estimator, never sets a hyperparameter on the wire, never
    touches a certificate, the sealed peek, or select-then-bound. Its entire output is a list of NAMES
    + prose. The downstream proposer (vfplatform/llm_moves.propose_moves) still resolves every move
    through the frozen catalog (harness.move_from_proposal), which CLAMPS params and DROPS unknown
    families. A family this surface "suggests" that is not in the catalog simply has no effect.
  * Family hints are INTERSECTED with the allowed catalog before they are returned, so a caller that
    feeds `family_hints` into a priority ordering can only ever reorder families the catalog already
    contains. Feature moves are likewise drawn from a fixed, enumerated FEATURE_MOVES registry.

DEGRADATION LADDER (each rung is strictly weaker context, never weaker safety):
  1. OFFLINE / no key / use_llm=False  -> a curated DETERMINISTIC RULE TABLE (KNOWLEDGE_RULES) keyed on
     measured profile+diagnosis signals. This is the default and the only path the tests exercise.
  2. LLM available (key + use_llm=True) -> ops.llm_propose with a schema-forced, NON_BINDING request;
     its output is VERIFIED back down to {catalog families} + {known feature moves} (anything else
     dropped) and UNIONED with the deterministic suggestions, so the LLM can only ever ADD in-vocab
     hints, never remove the deterministic floor or smuggle an out-of-vocab method.
  3. Web/dataset search (ToolSearch(WebSearch)) -> OPTIONAL literature note only, attached as prose to
     `notes`; it never changes the family/feature vocabulary. Injected by the caller via `web_search`
     (a callable) so this module has no hard network dependency and stays import-clean offline.

The rule table is curated from standard ML practice (sourced in comments at each rule), NOT reverse-
engineered from any target benchmark's answer. It only REORDERS / SUGGESTS within the bounded vocab.
"""
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in sys.path:
    sys.path.insert(0, _VF)

from vectorforge.llm_shell import ops

SURFACE = "vfplatform.knowledge_surface"
_TOOL_NAME = "emit_method_suggestions"
PROMPT_VERSION = "v1"


# ============================================================================== FEATURE MOVE REGISTRY
# The CLOSED vocabulary of feature-engineering moves this surface may suggest. Each is a named,
# parameter-light transform the PROPOSE step can choose to apply to the featurized design matrix
# BEFORE fitting (it is the caller's job to actually wire a transform; this surface only names it).
# Keeping this a fixed enum is the same bounding discipline the catalog uses for families: a hint can
# only ever be one of these, so an advisory layer can never introduce an unaudited transform.
FEATURE_MOVES = {
    "mutual_info_select": "Select top-k features by mutual information with the target (sklearn "
                          "SelectKBest/mutual_info_*). Helps when n_features is large vs n_train and "
                          "the target depends non-linearly on a sparse subset of features.",
    "variance_threshold": "Drop near-constant features (sklearn VarianceThreshold). Cheap denoiser "
                          "for wide tabular matrices with many dead columns.",
    "standardize": "Zero-mean/unit-variance scaling (StandardScaler). Required for distance- and "
                   "margin-based families (knn, svc_*, mlp) to behave; trees are scale-invariant.",
    "pca_whiten": "Linear dimensionality reduction (PCA) to decorrelate / compress a wide, "
                  "collinear feature matrix before a high-variance learner.",
    "interaction_terms": "Add pairwise interaction / low-degree polynomial features for a LINEAR "
                         "family when the target is non-linear but the feature count is small.",
    "tfidf_sublinear_tf": "Sublinear TF scaling (1+log(tf)) on the TF-IDF features; a standard text "
                          "tweak that dampens high-frequency terms (Manning et al., IR).",
    "tfidf_bigrams": "Extend the text featurizer to word bigrams; captures local phrase structure "
                     "that unigram TF-IDF misses (note: featurizer-level, advisory only here).",
    "class_weight_balance": "Reweight classes inversely to frequency for the fitted family; the "
                            "standard first response to a low minority-class recall.",
}


def known_feature_moves():
    """The closed set of feature-move names this surface may emit (read-only view for callers/tests)."""
    return set(FEATURE_MOVES.keys())


# ====================================================================================== SUGGESTION IO
@dataclass
class Suggestion:
    """One advisory item: prefer/avoid a catalog FAMILY, or apply a FEATURE MOVE. NON-BINDING."""
    kind: str            # "family" | "feature_move"
    name: str            # a catalog family name (intersected w/ allowed) OR a FEATURE_MOVES key
    weight: float        # advisory priority in [-1, 1]; >0 = prefer, <0 = de-prioritize
    reason: str          # one short, literature-/practice-grounded line

    def as_dict(self):
        return asdict(self)


@dataclass
class KnowledgeAdvice:
    """The surface's full advisory output. ALL fields are advisory; the proposer still clamps."""
    family_hints: list = field(default_factory=list)     # list[Suggestion(kind="family")]
    feature_moves: list = field(default_factory=list)    # list[Suggestion(kind="feature_move")]
    rationale: str = ""                                  # one-paragraph why, for logs / the LLM context
    notes: list = field(default_factory=list)            # optional literature/dataset notes (prose)
    source: str = "rules"                                # "rules" | "rules+llm" | "rules+web" | "rules+llm+web"

    def as_dict(self):
        return {"family_hints": [s.as_dict() for s in self.family_hints],
                "feature_moves": [s.as_dict() for s in self.feature_moves],
                "rationale": self.rationale, "notes": list(self.notes), "source": self.source}

    def ordered_families(self, catalog):
        """Convenience for the caller: catalog family names in DESCENDING advisory weight, then the
        rest of the catalog in its natural order. This is exactly the shape llm_moves._diagnosis_
        priority consumes -- so a caller can blend this advice into the existing priority WITHOUT this
        module importing the proposer (no cycle). Only families IN the catalog appear."""
        allowed = set(catalog or {})
        ranked = sorted([s for s in self.family_hints if s.name in allowed],
                        key=lambda s: -s.weight)
        head = [s.name for s in ranked]
        seen = set(head)
        return head + [f for f in (catalog or {}) if f not in seen]


# ================================================================================ DETERMINISTIC RULES
# Each rule: a PREDICATE over (profile, diagnosis) -> a list of Suggestions. Predicates read ONLY the
# trusted, platform-measured signals (shapes + the DIAGNOSE dict). Suggestions reference catalog family
# names and FEATURE_MOVES keys; out-of-catalog family hints are filtered out at the end, so a rule may
# safely name a family that a given task's catalog does not contain (e.g. a text family on a tabular
# task) without leaking it. Comments cite the practice each rule encodes.

# Family GROUPS by inductive bias, so rules can speak in capabilities, not brittle name lists. Names
# that aren't in the active catalog are dropped on the way out.
_NONLINEAR_TREE = ("hist_gbm", "random_forest", "extra_trees",
                   "hist_gbm_reg", "random_forest_reg", "extra_trees_reg")
_KERNEL = ("svc_rbf", "svr")
_NEURAL = ("mlp", "mlp_reg", "tfidf+mlp", "torch_mlp", "torch_cnn")
_LINEAR = ("logistic", "ridge", "lasso", "svc_linear",
           "tfidf+logistic", "tfidf+linear_svc", "tfidf+sgd_hinge", "tfidf+sgd_log")
_TEXT_NB = ("tfidf+multinomial_nb", "tfidf+complement_nb")


def _f(profile):
    p = profile or {}
    nf = p.get("n_features")
    ntr = p.get("n_train")
    ratio = (float(nf) / float(ntr)) if (nf and ntr) else None     # feature-to-sample ratio
    return {
        "kind": p.get("kind"),
        "task_type": p.get("task_type"),
        "is_text": p.get("kind") == "text",
        "is_reg": p.get("task_type") == "regression",
        "n_features": nf,
        "n_train": ntr,
        "n_classes": p.get("n_classes"),
        "fr_ratio": ratio,
        "high_dim": (ratio is not None and ratio >= 0.2) or (nf is not None and nf >= 200),
        "tiny": (ntr is not None and ntr < 300),
    }


def _d(diagnosis):
    d = diagnosis or {}
    return {
        "headroom": d.get("headroom"),
        "below_bar": d.get("below_bar"),
        "min_class_recall": d.get("min_class_recall"),
        "ece": d.get("ece"),
        "big_headroom": (d.get("headroom") or 0.0) > 0.02,
        "weak_class": (d.get("min_class_recall") is not None and d.get("min_class_recall") < 0.6),
        "miscalibrated": (d.get("ece") is not None and d.get("ece") > 0.10),
    }


def _rules(profile, diagnosis):
    """Curated rule table. Returns (family_suggestions, feature_suggestions, rationale_fragments)."""
    f = _f(profile)
    d = _d(diagnosis)
    fam, feat, why = [], [], []

    def addf(names, w, reason):
        for n in names:
            fam.append(Suggestion("family", n, w, reason))

    def addm(name, w, reason):
        feat.append(Suggestion("feature_move", name, w, reason))

    # R1 capacity: real headroom below the bar => escalate to non-linear capacity (trees first: robust,
    # few knobs, strong tabular default -- Grinsztajn et al. 2022, "trees beat DL on tabular").
    if d["big_headroom"] or d["below_bar"]:
        addf(_NONLINEAR_TREE, 0.9, "headroom below the bar -> gradient boosting / forests are the "
                                   "strongest low-tuning tabular families (Grinsztajn 2022).")
        addf(_KERNEL + _NEURAL, 0.5, "headroom -> kernel/neural capacity as a second-bias backstop.")
        why.append("validation sits below the bar with capacity headroom, so prefer higher-capacity "
                    "non-linear families over more linear sweeps")

    # R2 high-dim + nonlinear target => mutual-info feature selection + a non-linear learner (the
    # canonical 'p >> n with a sparse non-linear signal' recipe; Guyon & Elisseeff 2003).
    if f["high_dim"] and (d["big_headroom"] or d["below_bar"]):
        addm("mutual_info_select", 0.8, "high feature-to-sample ratio with headroom -> select the "
             "informative subset by mutual information before a non-linear fit (Guyon 2003).")
        addm("variance_threshold", 0.4, "wide matrix -> drop dead/near-constant columns first.")
        addf(_NONLINEAR_TREE, 0.6, "high-dim + non-linear headroom -> gradient boosting handles "
                                   "selected non-linear interactions well.")
        why.append("the feature-to-sample ratio is high, so mutual-information selection plus a "
                    "non-linear learner targets the sparse informative subset")

    # R3 high-dim + LINEAR-friendly (no headroom / small data) => keep it linear + regularize/compress.
    if f["high_dim"] and not (d["big_headroom"] or d["below_bar"]):
        addf(_LINEAR, 0.6, "high-dim but near the bar -> a regularized linear model is the bias-"
                           "appropriate, low-variance choice.")
        addm("pca_whiten", 0.4, "wide collinear matrix near the bar -> PCA compresses before fitting.")
        why.append("high-dim but near the bar favors a regularized linear model over added capacity")

    # R4 localized weakness: a class is being missed => rebalance + capacity/regularization.
    if d["weak_class"]:
        addm("class_weight_balance", 0.8, "low minority-class recall -> reweight classes inversely to "
             "frequency (standard first response to imbalance).")
        addf(_NONLINEAR_TREE, 0.5, "a missed class often needs the extra capacity to carve its region.")
        why.append("a single class has low recall, so rebalance class weights and add capacity to "
                    "recover the minority region")

    # R5 miscalibration => prefer calibratable / regularized fits; trees/SVM-margin tend to be over-
    # confident, logistic/NB are better calibrated out of the box.
    if d["miscalibrated"]:
        addf(("logistic", "tfidf+logistic", "tfidf+sgd_log") + _TEXT_NB, 0.6,
             "high ECE -> prefer probabilistically-calibrated families (logistic / NB) and regularize.")
        addm("standardize", 0.3, "scaling stabilizes the probability outputs of margin/neural fits.")
        why.append("the model is miscalibrated (ECE high), so prefer calibratable / regularized fits")

    # R6 text task => the TF-IDF-native zoo + standard text feature tweaks (advisory; featurizer is
    # harness-level so tfidf_bigrams is a hint, not an action here).
    if f["is_text"]:
        addf(_TEXT_NB + ("tfidf+complement_nb",), 0.5, "topic text -> Complement/Multinomial NB are "
             "strong cheap baselines; Complement NB corrects skew (Rennie 2003).")
        addf(("tfidf+linear_svc", "tfidf+logistic"), 0.5, "linear SVM / logistic on TF-IDF are classic "
             "strong text baselines (Joachims 1998).")
        addm("tfidf_sublinear_tf", 0.4, "sublinear TF dampens high-frequency terms (Manning IR).")
        if d["big_headroom"] or d["below_bar"]:
            addm("tfidf_bigrams", 0.4, "headroom on text -> bigrams capture phrase structure.")
        why.append("text task -> the TF-IDF-native families and standard text feature tweaks apply")

    # R7 distance/margin families need scaling; surface it whenever those families are in play AND we
    # are escalating capacity (so the caller knows to standardize before knn/svc/mlp).
    if (d["big_headroom"] or d["below_bar"]) and not f["is_text"]:
        addm("standardize", 0.3, "distance/margin/neural families (knn, svc, mlp) require feature "
             "scaling to behave; trees are scale-invariant so this is harmless for them.")

    # R8 tiny data => add a regularization-leaning interaction move for a small linear model, and warn
    # off the highest-variance neural family (advisory de-prioritization).
    if f["tiny"]:
        if not f["high_dim"]:
            addm("interaction_terms", 0.3, "small n with few features -> low-degree interactions let a "
                 "regularized linear model capture mild non-linearity without a high-variance learner.")
        addf(_NEURAL, -0.4, "very small training set -> de-prioritize high-variance neural families "
                            "(overfitting risk dominates the capacity gain).")
        why.append("the training set is small, so favor regularized low-variance models and de-"
                    "prioritize high-variance neural fits")

    # Default floor: if NOTHING fired (e.g. a clean above-bar profile), still give a sane ordering so
    # the surface never returns empty -- prefer the strong tabular default lightly.
    if not fam and not feat:
        addf(_NONLINEAR_TREE, 0.2, "no specific diagnosis -> the robust gradient-boosting default.")
        addf(_LINEAR, 0.1, "and a cheap linear baseline for contrast.")
        why.append("no specific lever diagnosed; default to a robust tabular ordering")

    return fam, feat, why


def _merge(suggestions):
    """Collapse duplicate suggestions for the same name. An EXPLICIT de-prioritization (negative weight)
    DOMINATES a generic positive backstop: if any rule says 'avoid X' (e.g. R8 warns off a high-variance
    neural family on tiny data), that avoid wins even if a broad capacity rule also listed X. Among same-
    sign suggestions we keep the strongest-magnitude one. This makes the rule table's sharp constraints
    (overfitting risk, miscalibration) override its broad defaults rather than the reverse."""
    by_name = {}
    for s in suggestions:
        by_name.setdefault(s.name, []).append(s)
    out = []
    for name, items in by_name.items():
        negatives = [s for s in items if s.weight < 0]
        pool = negatives if negatives else items
        # strongest magnitude in the governing pool (most-negative avoid, or most-positive prefer)
        winner = max(pool, key=lambda s: abs(s.weight))
        out.append(Suggestion(winner.kind, name, winner.weight, winner.reason))
    return sorted(out, key=lambda s: -s.weight)


def _merge_family(suggestions):
    return _merge(suggestions)


def _merge_moves(suggestions):
    return _merge(suggestions)


def _deterministic_advice(profile, diagnosis, catalog):
    """The OFFLINE rung: rule table -> filtered, merged KnowledgeAdvice. Always succeeds."""
    fam, feat, why = _rules(profile, diagnosis)
    allowed = set(catalog or {})
    # INTERSECT family hints with the allowed catalog (the bounding step). Feature moves are intersected
    # with the closed FEATURE_MOVES registry (they always are, by construction, but enforce it).
    fam = [s for s in fam if s.name in allowed] if allowed else fam
    feat = [s for s in feat if s.name in FEATURE_MOVES]
    fam = _merge_family(fam)
    feat = _merge_moves(feat)
    rationale = "; ".join(dict.fromkeys(why))      # dedup, preserve order
    return KnowledgeAdvice(family_hints=fam, feature_moves=feat, rationale=rationale,
                           notes=[], source="rules")


# ============================================================================================ LLM RUNG
_SYSTEM = (
    "You are a NON-BINDING method-knowledge advisor for an automated, certify-or-honest-fail model "
    "search. Given a dataset profile (shapes only) and the measured diagnosis of the current best "
    "model, suggest which model FAMILIES (from allowed_families ONLY) to prioritize and which named "
    "FEATURE MOVES (from allowed_feature_moves ONLY) to consider, with a one-line literature- or "
    "practice-grounded reason each. You are ADVISORY: you never build a model, set a hyperparameter, "
    "or decide anything. Any family or move outside the allowed lists will be DROPPED. Prefer fewer, "
    "well-justified suggestions over many. Give a positive weight in [0,1] for prefer, negative for "
    "de-prioritize.")

_SCHEMA = {
    "type": "object",
    "properties": {
        "family_hints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A family from allowed_families."},
                    "weight": {"type": "number", "description": "[-1,1]; >0 prefer, <0 de-prioritize."},
                    "reason": {"type": "string"},
                },
                "required": ["name"], "additionalProperties": False,
            },
        },
        "feature_moves": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A move from allowed_feature_moves."},
                    "weight": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["name"], "additionalProperties": False,
            },
        },
        "rationale": {"type": "string", "description": "One short paragraph: why, given the diagnosis."},
    },
    "required": ["family_hints"], "additionalProperties": False,
}


def _clamp_weight(w):
    try:
        w = float(w)
    except (TypeError, ValueError):
        return 0.5
    return max(-1.0, min(1.0, w))


def _llm_advice(profile, diagnosis, catalog, base, *, api_key, use_llm, cache_path, tenant_id, timeout):
    """The LLM rung: ask for in-vocab suggestions, VERIFY them down to {catalog} + {FEATURE_MOVES}, and
    UNION with the deterministic floor. The LLM can only ADD in-vocab hints. Never raises (ops handles
    every failure with the deterministic `fallback`, which returns the floor unchanged)."""
    allowed_fams = sorted(catalog or {})
    allowed_moves = sorted(FEATURE_MOVES)

    def _verify(raw):
        try:
            r = raw or {}
            fams, moves = [], []
            seen_f, seen_m = set(), set()
            for c in (r.get("family_hints") or []):
                nm = (c or {}).get("name")
                if nm in catalog and nm not in seen_f:
                    seen_f.add(nm)
                    fams.append({"name": nm, "weight": _clamp_weight((c or {}).get("weight", 0.5)),
                                 "reason": str((c or {}).get("reason", ""))[:240]})
            for c in (r.get("feature_moves") or []):
                nm = (c or {}).get("name")
                if nm in FEATURE_MOVES and nm not in seen_m:
                    seen_m.add(nm)
                    moves.append({"name": nm, "weight": _clamp_weight((c or {}).get("weight", 0.5)),
                                  "reason": str((c or {}).get("reason", ""))[:240]})
            if not fams and not moves:
                return False, None, "no in-vocab family/feature suggestions after filter"
            return True, {"family_hints": fams, "feature_moves": moves,
                          "rationale": str(r.get("rationale", ""))[:600]}, None
        except Exception as ex:  # noqa: BLE001
            return False, None, str(ex)[:160]

    def _fallback(_reason):
        # JSON post-image of the deterministic floor (so the union below is a no-op on failure).
        return {"family_hints": [s.as_dict() for s in base.family_hints],
                "feature_moves": [s.as_dict() for s in base.feature_moves],
                "rationale": base.rationale}

    req = ops.LLMRequest(
        surface=SURFACE, system=_SYSTEM,
        trusted_context={"profile": profile or {}, "diagnosis": diagnosis or {},
                         "allowed_families": allowed_fams,
                         "allowed_feature_moves": {k: FEATURE_MOVES[k] for k in allowed_moves}},
        untrusted_inputs={}, schema=_SCHEMA, tool_name=_TOOL_NAME,
        prompt_version=PROMPT_VERSION, force_tool=True, max_tokens=4000)

    p = ops.llm_propose(req, kind="NON_BINDING", verify=_verify, fallback=_fallback,
                        tenant_id=tenant_id, api_key=api_key, use_llm=use_llm,
                        cache_path=cache_path, timeout=timeout)
    val = p.value or {}

    # UNION the LLM's in-vocab hints with the deterministic floor (floor weights win on ties via merge's
    # max). The floor is never removed; the LLM can only raise a weight or add a new in-vocab hint.
    fam = list(base.family_hints) + [
        Suggestion("family", c["name"], c.get("weight", 0.5), c.get("reason", "") or "LLM hint")
        for c in val.get("family_hints", []) if c.get("name") in (catalog or {})]
    feat = list(base.feature_moves) + [
        Suggestion("feature_move", c["name"], c.get("weight", 0.5), c.get("reason", "") or "LLM hint")
        for c in val.get("feature_moves", []) if c.get("name") in FEATURE_MOVES]

    rationale = base.rationale
    extra = (val.get("rationale") or "").strip()
    if p.used_llm and extra:
        rationale = (rationale + " | LLM: " + extra) if rationale else ("LLM: " + extra)
    source = "rules+llm" if p.used_llm else "rules"
    return KnowledgeAdvice(family_hints=_merge_family(fam), feature_moves=_merge_moves(feat),
                           rationale=rationale, notes=list(base.notes), source=source)


# ============================================================================================ WEB RUNG
def _web_note(profile, diagnosis, web_search):
    """OPTIONAL literature/dataset note via a caller-injected `web_search(query)->str|list`. Returns a
    short prose note or None. NEVER changes the family/feature vocabulary; pure context for logs."""
    if web_search is None:
        return None
    f = _f(profile)
    q = ("best machine learning model and feature engineering for "
         + ("text classification" if f["is_text"]
            else "regression" if f["is_reg"] else "tabular classification")
         + (" high dimensional small sample" if f["high_dim"] else "")
         + (" class imbalance" if _d(diagnosis)["weak_class"] else ""))
    try:
        res = web_search(q)
    except Exception:  # noqa: BLE001  a flaky search must never affect advice
        return None
    if not res:
        return None
    if isinstance(res, (list, tuple)):
        res = "; ".join(str(x) for x in res[:3])
    return f"web[{q!r}]: {str(res)[:400]}"


# ================================================================================== PUBLIC ENTRY POINT
def suggest_methods(profile, diagnosis, catalog, *, use_llm=False, api_key=None,
                    cache_path=None, tenant_id="default", timeout=60.0, web_search=None):
    """The KNOWLEDGE SURFACE entry point. Returns a KnowledgeAdvice (ADVISORY, non-binding).

    profile      TRUSTED dataset profile: {kind, task_type, metric, n_train, n_val, n_features,
                 n_classes, threshold} (the same dict loop.py builds for the PROPOSE node).
    diagnosis    the round's measured DIAGNOSE signals {val, val_lb, headroom, ece, below_bar,
                 min_class_recall}. May be None (round 0).
    catalog      the per-task CATALOG (family -> CatalogEntry) the proposer is bounded to. Family hints
                 are INTERSECTED with this, so a hint can only ever name a family already in-bounds.
    use_llm      if True AND a key resolves, blend in an LLM rung (verified back into the vocab). Default
                 False so the surface is fully OFFLINE/deterministic by default.
    web_search   optional callable(query)->str|list for a literature note (e.g. a ToolSearch(WebSearch)
                 shim). Adds prose to `notes` only; never touches the vocabulary. Default None (offline).

    The output never reaches an estimator directly: the proposer still resolves every move through the
    frozen catalog, which clamps params and drops unknown families. This surface only REORDERS / SUGGESTS
    within the bounded vocabulary."""
    advice = _deterministic_advice(profile, diagnosis, catalog)

    if use_llm:
        advice = _llm_advice(profile, diagnosis, catalog, advice, api_key=api_key, use_llm=use_llm,
                             cache_path=cache_path, tenant_id=tenant_id, timeout=timeout)

    note = _web_note(profile, diagnosis, web_search)
    if note:
        advice.notes.append(note)
        advice.source = advice.source + "+web"

    return advice


# ======================================================================================== SELF-TEST
def _selftest():
    """Self-test : exercise a few profiles OFFLINE and print the advice."""
    clf = {"logistic", "svc_rbf", "random_forest", "extra_trees", "hist_gbm", "knn", "mlp", "svc_linear"}
    txt = {"tfidf+logistic", "tfidf+multinomial_nb", "tfidf+complement_nb", "tfidf+linear_svc",
           "tfidf+sgd_hinge", "tfidf+sgd_log", "tfidf+mlp"}
    catalogs = {"clf": {f: None for f in clf}, "text": {f: None for f in txt}}

    cases = [
        ("high-dim tabular, big headroom",
         {"kind": "tabular", "task_type": "binary", "n_train": 400, "n_features": 300, "n_classes": 2},
         {"headroom": 0.12, "below_bar": True, "min_class_recall": 0.71, "ece": 0.04}, "clf"),
        ("imbalanced low minority recall",
         {"kind": "tabular", "task_type": "binary", "n_train": 2000, "n_features": 40, "n_classes": 2},
         {"headroom": 0.01, "below_bar": False, "min_class_recall": 0.42, "ece": 0.05}, "clf"),
        ("miscalibrated, near bar",
         {"kind": "tabular", "task_type": "binary", "n_train": 5000, "n_features": 20, "n_classes": 2},
         {"headroom": 0.0, "below_bar": False, "min_class_recall": 0.88, "ece": 0.19}, "clf"),
        ("text with headroom",
         {"kind": "text", "task_type": "multiclass", "n_train": 3000, "n_features": 5000, "n_classes": 4},
         {"headroom": 0.08, "below_bar": True, "min_class_recall": 0.6, "ece": 0.07}, "text"),
    ]
    for name, prof, diag, cat in cases:
        adv = suggest_methods(prof, diag, catalogs[cat], use_llm=False)
        print(f"\n=== {name} [source={adv.source}] ===")
        print("  families:", [(s.name, s.weight) for s in adv.family_hints][:6])
        print("  moves   :", [(s.name, s.weight) for s in adv.feature_moves])
        print("  ordered :", adv.ordered_families(catalogs[cat])[:6])
        print("  why     :", adv.rationale[:160])


if __name__ == "__main__":
    _selftest()
