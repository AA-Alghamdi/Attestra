"""LLM-guided PROPOSE step (the diagram's "PROPOSE moves (LLM)") -- the recursive search's proposer.

This is the node the loop calls each round AFTER diagnosis: given the measured diagnosis signals, the
current leaderboard, and the dataset profile, propose the NEXT batch of candidate (family, hyperparameter)
configurations to try. It is built on vectorforge/llm_shell/ops.py `llm_propose`, so it inherits the whole
audited harness: schema-forced structured output, a replay cache, quarantine of untrusted inputs, and a
deterministic fallback for every failure mode.

NON-BINDING by construction (this is the integrity contract for a certify-or-honest-fail system):
  * The LLM may ONLY choose a family from the per-task CATALOG (harness.py) and hyperparameters WITHIN the
    catalog's safe ranges. The schema bounds the choice; the frozen resolve/clamp step (harness.resolve_family
    / move_from_proposal) maps each proposal to a real estimator ctor, CLAMPS every param to its safe range,
    and DROPS unknown families. An out-of-catalog family or out-of-range value can never reach an estimator.
  * The LLM NEVER sets the certificate, the decision, or the threshold theta. It proposes search moves only;
    selection stays on validation and the certificate is produced solely by the frozen certify path.
  * Already-tried configs are filtered out (`tried`), so each round explores NEW configurations.

Deterministic fallback (no Anthropic key / call fails / verify rejects): a GRID EXPANSION over the catalog
(harness._grid_configs). Even with NO LLM this proposes MANY distinct, diagnosis-prioritized configs (not 2),
so the recursive search keeps making real progress on the deterministic-only path.
"""
import os
import sys

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in sys.path:
    sys.path.insert(0, _VF)

from vectorforge.llm_shell import ops
from .harness import Move, move_from_proposal, _grid_configs

SURFACE = "vfplatform.propose_moves"
_TOOL_NAME = "emit_candidate_moves"
PROMPT_VERSION = "v1"

_SYSTEM = (
    "You are the PROPOSE node of an automated, certify-or-honest-fail model search. Given a dataset "
    "profile, the measured diagnosis of the current best model's validation error, and the leaderboard so "
    "far, propose the NEXT batch of candidate model configurations to try. You are NON-BINDING: you only "
    "PROPOSE candidates. You never set any certificate, decision, or threshold. Choose ONLY families from "
    "the provided allowed_catalog and ONLY hyperparameters within each family's stated ranges; any value "
    "outside a range will be clamped. Propose DISTINCT configs that are NOT already in tried_configs. Let "
    "the diagnosis steer you: if there is large headroom below the bar, propose higher-capacity families "
    "(boosting/forests/svc/mlp) and wider hyperparameter sweeps; if a single class has low recall, vary "
    "regularization and capacity; if the model is miscalibrated, prefer calibratable / regularized fits. "
    "Aim for 6-12 candidates that meaningfully cover the promising region of the search space.")


def _catalog_spec(catalog):
    """The TRUSTED, instruction-side description of the allowed families + ranges the LLM must stay within."""
    out = {}
    for fam, entry in catalog.items():
        ps = {}
        for name, spec in entry.params.items():
            if spec[0] == "choice":
                ps[name] = {"type": "choice", "allowed": list(spec[1])}
            else:
                ps[name] = {"type": spec[0], "min": spec[1], "max": spec[2]}
        out[fam] = {"params": ps}
    return out


_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "description": "The proposed next batch of model configurations to try.",
            "items": {
                "type": "object",
                "properties": {
                    "family": {"type": "string", "description": "A family name from allowed_catalog."},
                    "params": {"type": "object",
                               "description": "Hyperparameters within that family's ranges. "
                                              "Missing keys take safe defaults.",
                               "additionalProperties": True},
                },
                "required": ["family"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string", "description": "One short line: why these candidates, given the diagnosis."},
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def _diagnosis_priority(catalog, diagnosis):
    """Order the catalog families by what the MEASURED diagnosis makes relevant (so both the LLM context and
    the deterministic fallback explore the promising region first). Pure heuristic on measured signals; it
    only REORDERS the bounded catalog, it never invents a family or relaxes a bound."""
    d = diagnosis or {}
    headroom = d.get("headroom") or 0.0
    min_recall = d.get("min_class_recall")
    ece = d.get("ece")
    # capacity families first when there is headroom or a weak class; calibratable/regularized when miscalibrated
    capacity = ("torch_mlp", "torch_cnn", "hist_gbm", "random_forest", "extra_trees", "svc_rbf", "mlp",
                "hist_gbm_reg", "random_forest_reg", "extra_trees_reg", "svr", "mlp_reg",
                "tfidf+mlp")        # torch_* are in-catalog ONLY on a worker/GPU provider; tfidf+mlp is text
    linear = ("logistic", "svc_linear", "knn", "ridge", "lasso", "knn_reg", "tfidf+logistic",
              "tfidf+multinomial_nb", "tfidf+complement_nb", "tfidf+linear_svc",
              "tfidf+sgd_hinge", "tfidf+sgd_log")
    fams = list(catalog.keys())
    big = headroom > 0.02 or (min_recall is not None and min_recall < 0.6)
    if big:
        order = [f for f in capacity if f in catalog] + [f for f in linear if f in catalog]
    else:
        order = [f for f in linear if f in catalog] + [f for f in capacity if f in catalog]
    # stable, complete: append anything not listed
    seen = set(order)
    return order + [f for f in fams if f not in seen]


def _grid_fallback(catalog, diagnosis, tried, limit):
    """Deterministic grid-expansion proposer: enumerate the catalog's grids (Cartesian per family), in
    diagnosis-priority order, skipping already-tried configs, up to `limit` DISTINCT moves. With NO LLM this
    still proposes MANY distinct configs (not 2). Round-robins across families so a single huge grid does not
    crowd out diversity."""
    order = _diagnosis_priority(catalog, diagnosis)
    per_family = []          # list of (family, [configs not yet tried])
    for fam in order:
        cfgs = []
        for (f, params) in _grid_configs(catalog[fam]):
            m = move_from_proposal(catalog, f, params)
            if m is None or m.name in tried:
                continue
            cfgs.append((f, params, m))
        if cfgs:
            per_family.append(cfgs)
    moves, names = [], set()
    # round-robin: take one config from each family in priority order, repeat, until limit or exhausted
    i = 0
    while per_family and len(moves) < limit:
        progressed = False
        for cfgs in per_family:
            if i < len(cfgs):
                f, params, m = cfgs[i]
                if m.name not in names:
                    moves.append(m); names.add(m.name); progressed = True
                if len(moves) >= limit:
                    break
        if not progressed:
            break
        i += 1
    return moves


def propose_moves(profile, leaderboard_summary, diagnosis, task_type, metric, tried, *,
                  use_llm=True, api_key=None, catalog, cache_path=None, tenant_id="default",
                  limit=12, timeout=120.0):
    """The PROPOSE node. Returns (moves: list[Move], source: "llm"|"fallback").

    profile               trusted dataset summary (n_train/n_val, n_features, n_classes, kind, ...).
    leaderboard_summary   trusted summary of the runs so far (top families + val scores).
    diagnosis             the round's measured DIAGNOSE signals (headroom / min_class_recall / ece / below_bar).
    tried                 set of Move NAMES already executed -> never re-proposed.
    catalog               the per-task CATALOG (already filtered to the provider's runnable families).

    The returned moves are FROZEN-RESOLVED: each is a real estimator ctor with CLAMPED params (unknown
    families dropped). Deterministic grid-expansion fallback on any LLM failure -> many distinct configs."""
    tried = set(tried or ())
    if not catalog:
        return [], "fallback"

    def _verify(raw):
        """Frozen verify: take the LLM's candidates, resolve+clamp each through the catalog, drop unknown
        families and already-tried/duplicate configs. Returns the bounded post-image (a list of {family,
        params, move_name}) -- buildable and within-bounds by construction."""
        try:
            cands = (raw or {}).get("candidates") or []
            bounded, names = [], set()
            for c in cands:
                fam = (c or {}).get("family")
                params = (c or {}).get("params") or {}
                m = move_from_proposal(catalog, fam, params)
                if m is None or m.name in tried or m.name in names:
                    continue
                names.add(m.name)
                # store the clamped post-image (the move's single family carries clamped params)
                bounded.append({"family": catalog[fam].family, "move_name": m.name,
                                "params": {k: v for k, v in m.families[0][2].items() if k != "family"}})
                if len(bounded) >= limit:
                    break
            if not bounded:
                return False, None, "no in-catalog, novel candidates after clamp/dedup"
            return True, {"candidates": bounded}, None
        except Exception as ex:  # noqa: BLE001
            return False, None, str(ex)[:160]

    def _fallback(_reason):
        # JSON-serializable post-image for the cache shape parity (the loop rebuilds Moves below either way)
        moves = _grid_fallback(catalog, diagnosis, tried, limit)
        return {"candidates": [{"family": m.families[0][2].get("family", m.name),
                                "move_name": m.name,
                                "params": {k: v for k, v in m.families[0][2].items() if k != "family"}}
                               for m in moves]}

    req = ops.LLMRequest(
        surface=SURFACE, system=_SYSTEM,
        trusted_context={"task_type": task_type, "metric": metric, "profile": profile,
                         "diagnosis": diagnosis or {}, "leaderboard": leaderboard_summary or {},
                         "allowed_catalog": _catalog_spec(catalog),
                         "tried_configs": sorted(tried)[:200],
                         "max_candidates": limit},
        untrusted_inputs={},                       # nothing user-controlled here; all signals are platform-measured
        schema=_SCHEMA, tool_name=_TOOL_NAME, prompt_version=PROMPT_VERSION,
        force_tool=True, max_tokens=40000)

    p = ops.llm_propose(req, kind="NON_BINDING", verify=_verify, fallback=_fallback,
                        tenant_id=tenant_id, api_key=api_key, use_llm=use_llm,
                        cache_path=cache_path, timeout=timeout)

    # Rebuild real Moves from the bounded post-image (works for both the LLM and the fallback shape). Every
    # config is re-resolved through the catalog, so the loop only ever sees clamped, buildable moves.
    moves, names = [], set()
    for c in (p.value or {}).get("candidates", []):
        m = move_from_proposal(catalog, c.get("family"), c.get("params") or {})
        if m is None or m.name in tried or m.name in names:
            continue
        names.add(m.name)
        moves.append(m)
    source = "llm" if (p.used_llm and moves) else "fallback"
    if not moves:
        # the LLM produced nothing usable AND the (cached) fallback shape was empty -> regenerate the grid
        moves = _grid_fallback(catalog, diagnosis, tried, limit)
        source = "fallback"
    return moves, source
