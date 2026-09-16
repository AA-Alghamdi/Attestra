"""Tests for frontier.intelligence -- search intelligence layer."""
import numpy as np
import pytest

from frontier.intelligence import (
    IntelligenceState,
    LiteratureContext,
    ASHAState,
    FeatureEngineeringProposer,
    EnsembleProposer,
    PromptEvolution,
    init_intelligence,
    enrich_round_context,
    get_intelligence_proposals,
    record_intelligence_outcome,
    record_repair_failure,
    literature_scout,
    multi_turn_diagnose,
    adaptive_subsample,
    knowledge_readback,
    enrich_context_with_literature,
    enrich_context_with_feature_hints,
    enrich_context_with_evolution,
    init_asha,
    asha_proposals,
    _config_to_code,
    _hp_space_for_family,
)
from frontier.task import Task
from frontier.program import Program


def _make_task(kind="classification", n_features=10, metric="accuracy", theta=0.8, n_samples=100):
    """Helper to create Task with dummy data."""
    X = np.random.randn(n_samples, n_features)
    if kind == "classification":
        y = np.random.randint(0, 3, n_samples).astype(str)
    else:
        y = np.random.randn(n_samples)
    return Task(X=X, y=y, kind=kind, theta=theta, metric=metric)


# ============================================================================
# Literature Scout
# ============================================================================

class TestLiteratureScout:
    def test_literature_scout_degrades_offline(self):
        ctx = literature_scout("classify tumors", online=False)
        assert isinstance(ctx, LiteratureContext)
        assert ctx.n_findings >= 0

    def test_literature_scout_with_llm_fallback(self):
        responses = []
        def mock_llm(prompt):
            responses.append(prompt)
            return "Use XGBoost with early stopping. Use SMOTE for imbalance."

        ctx = literature_scout("classify tumors with imbalance", llm_client=mock_llm)
        assert ctx.n_findings >= 0

    def test_enrich_context_with_literature(self):
        ctx = {"task_kind": "classification", "n_features": 10}
        lit = LiteratureContext(
            n_findings=3,
            techniques=["XGBoost with early stopping", "SMOTE for imbalance"],
            architectures=["RandomForest", "GradientBoosting"],
            key_insights=["Key: ensemble methods work best"],
        )
        result = enrich_context_with_literature(ctx, lit)
        assert "literature" in result
        assert result["literature"]["n_findings"] == 3
        assert "llm_guidance" in result
        assert "LITERATURE" in result["llm_guidance"]

    def test_empty_literature_no_enrichment(self):
        ctx = {"task_kind": "classification"}
        lit = LiteratureContext()
        result = enrich_context_with_literature(ctx, lit)
        assert "literature" not in result

    def test_key_insights_reach_prompt(self):
        # Retrieved paper abstracts (not just keyword motifs) must reach the LLM prompt.
        ctx = {"task_kind": "classification"}
        lit = LiteratureContext(
            n_findings=2,
            techniques=[],
            key_insights=[
                "[arxiv] DANets: Deep Abstract Networks for Tabular Data: abstract...",
                "[arxiv] Generalized Oversampling for Imbalanced datasets: abstract...",
            ],
        )
        result = enrich_context_with_literature(ctx, lit)
        guidance = result["llm_guidance"]
        assert "DANets" in guidance
        assert "Generalized Oversampling" in guidance
        assert "research surface" in guidance

    def test_scout_query_is_keyword_reduced(self, monkeypatch):
        # A long natural-language goal must be reduced to salient keywords before
        # it reaches the retrieval backend; the raw phrase ranks poorly on arXiv's
        # `all:` endpoint and returns off-domain (e.g. particle-physics) papers.
        captured = {}

        class _FakeScout:
            def __init__(self, problem, **kwargs):
                captured["problem"] = problem

            def summary(self):
                return {"n_findings": 0, "motifs": [], "backbones": [],
                        "findings": []}

        import vfplatform.literature as _lit
        monkeypatch.setattr(_lit, "LiteratureScout", _FakeScout)

        goal = "forecast a noisy univariate time series with regime shifts"
        literature_scout(goal, online=True)

        q = captured["problem"]
        # filler/stopwords dropped, salient terms preserved
        assert q != goal
        for filler in (" a ", " with "):
            assert filler not in f" {q} "
        for kw in ("forecast", "time", "series", "regime"):
            assert kw in q


# ============================================================================
# ASHA Hyperparameter Search
# ============================================================================

class TestASHA:
    def _make_record(self, label, ok, val_score):
        class _R:
            pass
        r = _R()
        r.label = label
        r.ok = ok
        r.val_score = val_score
        return r

    def test_init_asha_requires_round_1(self):
        history = [self._make_record("rf_0", True, 0.85)]
        task = _make_task()
        state = init_asha(history, task, 0)
        assert state.best_family is None

    def test_init_asha_identifies_best_family(self):
        history = [
            self._make_record("rf_0", True, 0.85),
            self._make_record("hist_gbm_1", True, 0.90),
            self._make_record("logreg_2", True, 0.75),
        ]
        task = _make_task()
        state = init_asha(history, task, 1)
        assert state.best_family == "hist_gbm"

    def test_hp_space_for_known_family(self):
        space = _hp_space_for_family("rf", "classification")
        assert "n_estimators" in space
        assert "max_depth" in space

    def test_hp_space_for_unknown_family(self):
        space = _hp_space_for_family("unknown_model", "classification")
        assert space == {}

    def test_config_to_code_valid(self):
        config = {"n_estimators": 200, "max_depth": 10}
        code = _config_to_code("rf", config, "classification")
        assert "RandomForestClassifier" in code
        assert "build_estimator" in code
        assert "n_estimators=200" in code

    def test_config_to_code_unknown_family(self):
        code = _config_to_code("unknown", {}, "classification")
        assert code == ""

    def test_init_asha_builds_tpe_sampler(self):
        # Guards the real vfplatform.hp_search API (TPE, tuple param_specs):
        # a prior mismatch (TPESampler/.sample()/dict-spec) silently left tpe=None
        # so ASHA-HP proposals never fired.
        history = [
            self._make_record("hist_gbm_0", True, 0.90),
            self._make_record("rf_1", True, 0.85),
        ]
        state = init_asha(history, _make_task(), 1)
        assert state.best_family == "hist_gbm"
        assert state.tpe is not None

    def test_asha_proposals_fire(self):
        history = [
            self._make_record("hist_gbm_0", True, 0.90),
            self._make_record("rf_1", True, 0.85),
        ]
        task = _make_task()
        state = init_asha(history, task, 1)
        programs = asha_proposals(state, task, n=3)
        assert len(programs) == 3
        for p in programs:
            assert p.source == "asha"
            assert p.label.startswith("asha_hist_gbm_")
            assert "build_estimator" in p.code


# ============================================================================
# Feature Engineering
# ============================================================================

class TestFeatureEngineering:
    def test_feature_proposer_generates_programs(self):
        fp = FeatureEngineeringProposer(n_features=20, kind="classification")
        ctx = {"task_kind": "classification", "n_features": 20, "best_label": "rf_0"}
        programs = fp.propose(ctx)
        assert len(programs) > 0
        for p in programs:
            assert p.source == "feature_eng"
            assert "build_estimator" in p.code

    def test_feature_proposer_only_proposes_once(self):
        fp = FeatureEngineeringProposer(n_features=20, kind="classification")
        ctx = {"task_kind": "classification", "n_features": 20}
        p1 = fp.propose(ctx)
        p2 = fp.propose(ctx)
        assert len(p1) > 0
        assert len(p2) == 0

    def test_feature_proposer_skips_poly_for_high_dim(self):
        fp = FeatureEngineeringProposer(n_features=100, kind="classification")
        ctx = {"task_kind": "classification", "n_features": 100}
        programs = fp.propose(ctx)
        labels = [p.label for p in programs]
        assert not any("poly2" in l for l in labels)

    def test_enrich_context_with_feature_hints(self):
        ctx = {"task_kind": "classification"}
        task = _make_task(n_features=60)
        result = enrich_context_with_feature_hints(ctx, task)
        assert "feature_hints" in result
        assert "llm_guidance" in result
        assert "PCA" in result["llm_guidance"]


# ============================================================================
# Knowledge Read-Back
# ============================================================================

class TestKnowledgeReadback:
    def test_readback_without_fingerprint(self):
        ctx = {"task_kind": "classification"}
        result = knowledge_readback(ctx)
        assert "kb_hints" not in result

    def test_readback_handles_missing_module(self):
        ctx = {"task_kind": "classification", "task_fingerprint": [0.1, 0.2, 0.3]}
        result = knowledge_readback(ctx, kb_path="/nonexistent/path")
        assert isinstance(result, dict)


# ============================================================================
# Multi-Turn Diagnosis
# ============================================================================

class TestMultiTurnDiagnosis:
    def _make_diag(self):
        from frontier.diagnosis import Diagnosis, FamilyStat
        return Diagnosis(
            round_index=1, n_ok=5, n_fail=3, n_candidates=8,
            best_label="rf_0", best_score=0.85,
            family_rank=[FamilyStat("rf", 5, 3, 0.85, 0.80)],
            dominant_error_kind="runtime", dominant_error_share=0.6,
            axis_lift={}, plateau=False, plateau_span=0,
            residual_summary="", directives={}, reason="testing",
        )

    def test_multi_turn_without_llm(self):
        diag = self._make_diag()
        task = _make_task()
        result = multi_turn_diagnose(diag, task, [])
        assert not result.available

    def test_multi_turn_with_prior_failures(self):
        diag = self._make_diag()
        task = _make_task()
        responses = []
        def mock_llm(prompt):
            responses.append(prompt)
            return (
                "ROOT CAUSES: Overfitting on small data.\n"
                "TARGETED IMPROVEMENTS: Use regularization.\n"
                "ARCHITECTURE RECOMMENDATIONS: Ridge with alpha tuning.\n"
                "REPAIR STRATEGIES: Add StandardScaler."
            )

        from frontier.llm_diagnosis import LLMDiagnosis
        prior = LLMDiagnosis(
            available=True, targeted_guidance="Try more trees",
            code_suggestions=[], confusion_analysis=None,
            residual_analysis="", root_causes=["Underfitting"],
            recommended_architectures=["RF"], repair_strategies=["Add trees"],
        )
        failures = [{"error_kind": "runtime", "error": "fit failed"}]

        result = multi_turn_diagnose(
            diag, task, [], llm_client=mock_llm,
            prior_diagnosis=prior, prior_repair_failures=failures,
        )
        assert result.available
        assert len(responses) == 2  # initial + refinement


# ============================================================================
# Ensemble Proposer
# ============================================================================

class TestEnsembleProposer:
    def test_ensemble_requires_round_1(self):
        ep = EnsembleProposer()
        ctx = {"round": 0, "task_kind": "classification", "best_label": "rf_0"}
        assert ep.propose(ctx) == []

    def test_ensemble_generates_multiple_variants(self):
        ep = EnsembleProposer()
        ctx = {"round": 1, "task_kind": "classification", "best_label": "rf_0", "best_score": 0.85}
        programs = ep.propose(ctx)
        assert len(programs) == 4  # B1: voting_diverse, stacking_focused, voting_weighted, stacking_deep
        sources = {p.source for p in programs}
        assert sources == {"ensemble"}
        labels = {p.label for p in programs}
        assert any("voting_diverse" in l for l in labels)
        assert any("stacking_focused" in l for l in labels)
        assert any("voting_weighted" in l for l in labels)
        assert any("stacking_deep" in l for l in labels)

    def test_ensemble_deduplicates(self):
        ep = EnsembleProposer()
        ctx = {"round": 1, "task_kind": "classification", "best_label": "rf_0", "best_score": 0.85}
        p1 = ep.propose(ctx)
        p2 = ep.propose(ctx)  # same round
        assert len(p1) == 4
        assert len(p2) == 0

    def test_ensemble_regression(self):
        ep = EnsembleProposer()
        ctx = {"round": 1, "task_kind": "regression", "best_label": "rf_0", "best_score": 0.7}
        programs = ep.propose(ctx)
        assert len(programs) == 4
        for p in programs:
            assert "Regressor" in p.code or "Ridge" in p.code

    def test_ensemble_adapts_to_history(self):
        """B1: Ensemble composition adapts based on which families scored well."""
        ep = EnsembleProposer()
        # Feed history: hist_gbm consistently best
        ep.record_family_score("hist_gbm_0", 0.95, "classification")
        ep.record_family_score("hist_gbm_1", 0.93, "classification")
        ep.record_family_score("rf_0", 0.85, "classification")
        ep.record_family_score("logreg_0", 0.70, "classification")

        ctx = {"round": 2, "task_kind": "classification", "best_label": "hist_gbm_0"}
        programs = ep.propose(ctx)
        assert len(programs) == 4
        # Check provenance: ranked families should reflect history
        for p in programs:
            families = p.provenance.get("families", [])
            if families:
                assert families[0] == "hist_gbm"  # best-performing should be first


# ============================================================================
# Adaptive Subsampling
# ============================================================================

class TestAdaptiveSubsample:
    def test_no_subsample_small_data(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_sub, y_sub, was_sub = adaptive_subsample(X, y, max_rows=200)
        assert not was_sub
        assert X_sub.shape[0] == 100

    def test_subsample_large_data(self):
        X = np.random.randn(50000, 10)
        y = np.random.randint(0, 3, 50000)
        # Use last round so progressive scaling gives full max_rows
        X_sub, y_sub, was_sub = adaptive_subsample(
            X, y, max_rows=5000, round_idx=2, total_rounds=3)
        assert was_sub
        assert abs(X_sub.shape[0] - 5000) <= 10  # stratified sampling may round
        assert len(y_sub) == X_sub.shape[0]

    def test_progressive_subsample_scales_up(self):
        """B2: Early rounds use less data, later rounds use more."""
        X = np.random.randn(50000, 10)
        y = np.random.randint(0, 3, 50000)
        # Round 0 of 3: should use ~50% of max_rows
        X0, y0, s0 = adaptive_subsample(X, y, max_rows=10000, round_idx=0, total_rounds=3)
        # Round 2 (last) of 3: should use full max_rows
        X2, y2, s2 = adaptive_subsample(X, y, max_rows=10000, round_idx=2, total_rounds=3)
        assert s0 and s2
        # Last round should have more data than first
        assert X2.shape[0] >= X0.shape[0]

    def test_subsample_preserves_classes(self):
        X = np.random.randn(10000, 5)
        y = np.array([0]*9000 + [1]*900 + [2]*100)
        X_sub, y_sub, was_sub = adaptive_subsample(X, y, max_rows=1000)
        assert was_sub
        unique = set(y_sub)
        assert 0 in unique
        assert 1 in unique
        assert 2 in unique


# ============================================================================
# Prompt Evolution
# ============================================================================

class TestPromptEvolution:
    def test_empty_evolution(self):
        evo = PromptEvolution()
        assert evo.evolution_block() == ""

    def test_records_and_evolves_with_directives(self):
        """B3: Evolution produces EMPHASIZE/AVOID directives from scored data."""
        evo = PromptEvolution()
        prog_ok = Program(source="seed", label="rf_0", code="...")
        prog_fail = Program(source="seed", label="logreg_0", code="...")

        record_intelligence_outcome(
            IntelligenceState(prompt_evolution=evo), prog_ok, 0.85, True)
        record_intelligence_outcome(
            IntelligenceState(prompt_evolution=evo), prog_fail, None, False)
        # Need 2+ failures for AVOID directive
        prog_fail2 = Program(source="seed", label="logreg_1", code="...")
        record_intelligence_outcome(
            IntelligenceState(prompt_evolution=evo), prog_fail2, None, False)

        block = evo.evolution_block()
        # B3: should produce directive-style output with scored data
        assert "EMPHASIZE" in block or "WORKED" in block
        assert "AVOID" in block or "FAILED" in block

    def test_enrich_context_with_evolution(self):
        evo = PromptEvolution()
        evo.successful_techniques = ["rf_0"]
        evo.failed_techniques = ["logreg_0"]
        ctx = {}
        result = enrich_context_with_evolution(ctx, evo)
        assert "llm_guidance" in result
        assert "EVOLUTION" in result["llm_guidance"] or "PROMPT" in result["llm_guidance"]

    def test_template_rewriting(self):
        """B3: PromptEvolution actually rewrites LLM prompt templates."""
        evo = PromptEvolution()
        # Record several outcomes to build scored data
        for label, score in [("rf_0", 0.90), ("rf_1", 0.92), ("hist_gbm_0", 0.88)]:
            prog = Program(source="seed", label=label, code="...")
            evo.record_outcome(prog, score, True)
        for label in ["logreg_0", "logreg_1"]:
            prog = Program(source="seed", label=label, code="...")
            evo.record_outcome(prog, None, False)

        template = "You are an ML expert. Generate a sklearn pipeline."
        rewritten = evo.rewrite_template(template)
        assert rewritten != template  # template was actually modified
        assert "EVOLUTION" in rewritten
        assert "Generate" in rewritten  # original instruction preserved

    def test_evolution_score_trends(self):
        """B3: Evolution detects improving/degrading/plateauing trends."""
        evo = PromptEvolution()
        # Simulate improving scores
        for i, score in enumerate([0.70, 0.72, 0.75, 0.80, 0.85, 0.88]):
            prog = Program(source="seed", label=f"rf_{i}", code="...")
            evo.record_outcome(prog, score, True)
        block = evo.evolution_block()
        assert "TREND" in block
        assert "Improving" in block or "current direction" in block.lower()


# ============================================================================
# Integration: init + enrich + proposals
# ============================================================================

class TestIntelligenceIntegration:
    def test_init_intelligence(self):
        task = _make_task()
        state = init_intelligence("classify tumors", task)
        assert isinstance(state, IntelligenceState)
        assert state.feature_proposer is not None
        assert state.ensemble_proposer is not None
        assert state.prompt_evolution is not None

    def test_enrich_round_context(self):
        task = _make_task()
        state = init_intelligence("classify tumors", task)
        ctx = {"task_kind": "classification", "n_features": 10, "round": 0}
        result = enrich_round_context(ctx, state, task, 0, [])
        assert "feature_hints" in result

    def test_get_intelligence_proposals(self):
        task = _make_task(n_features=20)
        state = init_intelligence("classify tumors", task)
        ctx = {
            "task_kind": "classification", "n_features": 20,
            "round": 0, "best_label": "rf_0",
        }
        proposals = get_intelligence_proposals(state, ctx, task)
        assert len(proposals) > 0
        sources = {p.source for p in proposals}
        assert "feature_eng" in sources

    def test_repair_failure_tracking(self):
        state = IntelligenceState()
        record_repair_failure(state, "runtime", "fit failed")
        record_repair_failure(state, "timeout", "exceeded 60s")
        assert len(state.repair_failures) == 2
        assert state.repair_failures[0]["error_kind"] == "runtime"

    def test_full_round_cycle(self):
        task = _make_task(n_features=15)
        state = init_intelligence("classify data", task)

        # Round 0: feature proposals
        ctx = {"task_kind": "classification", "n_features": 15, "round": 0, "best_label": ""}
        enrich_round_context(ctx, state, task, 0, [])
        proposals = get_intelligence_proposals(state, ctx, task)
        assert len(proposals) > 0

        # Record outcomes
        for p in proposals:
            record_intelligence_outcome(state, p, 0.8, True)

        # Round 1: should get ensemble proposals too
        ctx = {"task_kind": "classification", "n_features": 15,
               "round": 1, "best_label": "rf_0", "best_score": 0.85}
        enrich_round_context(ctx, state, task, 1, [])
        proposals_r1 = get_intelligence_proposals(state, ctx, task)
        sources = {p.source for p in proposals_r1}
        assert "ensemble" in sources
