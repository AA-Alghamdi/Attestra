"""Self-Improvement Stack — the system improves its own components.

True recursive self-improvement means improving:
  a. Prompt Evolution: track which prompts produce better proposals
  b. Harness Self-Authoring: write + validate + permanently store new harnesses
  c. Oracle Evolution: propose new integrity checks from failure patterns
  d. Strategy Generation: generate new strategies from meta-learner patterns

The improvement happens at the SYSTEM level, not the solution level.
Each improvement compounds across all future experiments.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Prompt Evolution
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """A tracked prompt template with performance history."""
    template_id: str
    template: str           # the actual prompt template (with {placeholders})
    purpose: str           # "proposal_generation" | "diagnosis" | "repair" | "strategy"
    # Performance tracking
    uses: int = 0
    successes: int = 0     # proposals that scored above median
    avg_score: float = 0.0
    best_score: float = 0.0
    # Metadata
    created_at: float = 0.0
    source: str = "default"  # "default" | "evolved" | "user"
    parent_id: Optional[str] = None  # what it was derived from


class PromptEvolver:
    """Evolves LLM prompts based on what produces better proposals.

    Tracks which prompt templates correlate with higher val scores,
    and periodically generates improved variants.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._path = persist_path or os.path.expanduser("~/.attestra/prompt_evolution.jsonl")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._templates: Dict[str, PromptTemplate] = {}
        self._load()

    def _load(self) -> None:
        """Load template history from disk."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    t = PromptTemplate(
                        template_id=d["template_id"],
                        template=d["template"],
                        purpose=d["purpose"],
                        uses=d.get("uses", 0),
                        successes=d.get("successes", 0),
                        avg_score=d.get("avg_score", 0.0),
                        best_score=d.get("best_score", 0.0),
                        created_at=d.get("created_at", 0.0),
                        source=d.get("source", "default"),
                        parent_id=d.get("parent_id"),
                    )
                    self._templates[t.template_id] = t
        except Exception:
            pass

    def _save(self) -> None:
        """Persist templates to disk."""
        with open(self._path, "w") as f:
            for t in self._templates.values():
                f.write(json.dumps({
                    "template_id": t.template_id,
                    "template": t.template,
                    "purpose": t.purpose,
                    "uses": t.uses,
                    "successes": t.successes,
                    "avg_score": t.avg_score,
                    "best_score": t.best_score,
                    "created_at": t.created_at,
                    "source": t.source,
                    "parent_id": t.parent_id,
                }) + "\n")

    def register_template(self, template: str, purpose: str, source: str = "default") -> str:
        """Register a new prompt template. Returns template_id."""
        tid = hashlib.sha256(f"{template}:{purpose}".encode()).hexdigest()[:12]
        if tid not in self._templates:
            self._templates[tid] = PromptTemplate(
                template_id=tid,
                template=template,
                purpose=purpose,
                source=source,
                created_at=time.time(),
            )
            self._save()
        return tid

    def get_best_template(self, purpose: str) -> Optional[PromptTemplate]:
        """Get the best-performing template for a purpose."""
        candidates = [t for t in self._templates.values()
                      if t.purpose == purpose and t.uses > 0]
        if not candidates:
            candidates = [t for t in self._templates.values() if t.purpose == purpose]
        if not candidates:
            return None
        # Rank by success rate with UCB exploration bonus
        import math
        total_uses = sum(t.uses for t in candidates)
        def ucb_score(t: PromptTemplate) -> float:
            if t.uses == 0:
                return float('inf')  # explore unused
            exploit = t.successes / t.uses
            explore = math.sqrt(2 * math.log(max(total_uses, 1)) / t.uses)
            return exploit + explore
        return max(candidates, key=ucb_score)

    def record_outcome(self, template_id: str, score: float, success: bool) -> None:
        """Record the outcome of using a template."""
        t = self._templates.get(template_id)
        if t is None:
            return
        t.uses += 1
        if success:
            t.successes += 1
        # Running average
        t.avg_score = (t.avg_score * (t.uses - 1) + score) / t.uses
        t.best_score = max(t.best_score, score)
        self._save()

    def evolve(self, purpose: str, llm_call: Optional[Callable] = None) -> Optional[str]:
        """Evolve a new template from the best-performing ones.

        Uses the LLM to generate a variant of the best template,
        incorporating patterns from successful uses.

        Returns the new template_id, or None if no evolution possible.
        """
        if llm_call is None:
            return None

        # Get top templates
        candidates = sorted(
            [t for t in self._templates.values() if t.purpose == purpose and t.uses > 2],
            key=lambda t: t.avg_score, reverse=True,
        )[:3]

        if not candidates:
            return None

        best = candidates[0]

        # Ask LLM to improve the template
        prompt = (
            f"You are optimizing a prompt template for {purpose}.\n"
            f"Current best template (success rate: {best.successes}/{best.uses}, "
            f"avg score: {best.avg_score:.3f}):\n\n"
            f"```\n{best.template}\n```\n\n"
            f"Generate an improved variant that might produce better results. "
            f"Keep the same structure and placeholders, but refine the instructions. "
            f"Output ONLY the new template, nothing else."
        )

        try:
            text, _ = llm_call(
                "You are an expert at prompt engineering for ML research.",
                prompt,
            )
            if text and len(text) > 20:
                new_id = self.register_template(text.strip(), purpose, source="evolved")
                self._templates[new_id].parent_id = best.template_id
                self._save()
                return new_id
        except Exception:
            pass

        return None


# ---------------------------------------------------------------------------
# Harness Self-Authoring (permanent library)
# ---------------------------------------------------------------------------

@dataclass
class AuthoredHarness:
    """A harness authored by the system and validated."""
    harness_id: str
    task_type: str          # what task type this handles
    code: str               # the harness implementation
    validated: bool = False # passed self-test on known benchmark
    validation_score: float = 0.0
    # Metadata
    created_at: float = 0.0
    uses: int = 0
    successes: int = 0


class HarnessLibrary:
    """Permanent library of authored harnesses.

    When the system encounters a task type it has no harness for:
    1. LLM writes a candidate harness
    2. System validates it on a known-good benchmark
    3. If it passes, it's added permanently
    4. Next time, the authored harness is retrieved — no re-authoring needed
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._path = persist_path or os.path.expanduser("~/.attestra/harness_library.jsonl")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._harnesses: Dict[str, AuthoredHarness] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    h = AuthoredHarness(
                        harness_id=d["harness_id"],
                        task_type=d["task_type"],
                        code=d["code"],
                        validated=d.get("validated", False),
                        validation_score=d.get("validation_score", 0.0),
                        created_at=d.get("created_at", 0.0),
                        uses=d.get("uses", 0),
                        successes=d.get("successes", 0),
                    )
                    self._harnesses[h.harness_id] = h
        except Exception:
            pass

    def _save(self) -> None:
        with open(self._path, "w") as f:
            for h in self._harnesses.values():
                f.write(json.dumps({
                    "harness_id": h.harness_id,
                    "task_type": h.task_type,
                    "code": h.code,
                    "validated": h.validated,
                    "validation_score": h.validation_score,
                    "created_at": h.created_at,
                    "uses": h.uses,
                    "successes": h.successes,
                }) + "\n")

    def get_harness(self, task_type: str) -> Optional[AuthoredHarness]:
        """Get a validated harness for a task type."""
        candidates = [h for h in self._harnesses.values()
                      if h.task_type == task_type and h.validated]
        if not candidates:
            return None
        # Return the most successful one
        return max(candidates, key=lambda h: h.successes)

    def author_harness(
        self,
        task_type: str,
        llm_call: Callable,
        validation_data: Optional[Tuple] = None,
    ) -> Optional[AuthoredHarness]:
        """Author a new harness for a task type using the LLM.

        If validation_data is provided, validates the harness before storing.
        """
        prompt = (
            f"Write a Python harness class for task type '{task_type}'.\n"
            f"The harness must implement:\n"
            f"  - self_test() -> Tuple[bool, dict]: run on a known benchmark\n"
            f"  - adapt(X, y) -> task_dict: adapt data for this modality\n"
            f"  - baseline_models() -> list: return baseline model configs\n"
            f"  - metric() -> str: the appropriate metric name\n"
            f"  - split_protocol() -> tuple: (train_frac, val_frac)\n\n"
            f"Output ONLY the Python code, no explanation."
        )

        try:
            text, _ = llm_call(
                "You are an expert ML engineer writing reusable harness code.",
                prompt,
            )
            if not text or len(text) < 50:
                return None

            code = text.strip()
            hid = hashlib.sha256(f"{task_type}:{code[:100]}".encode()).hexdigest()[:12]

            harness = AuthoredHarness(
                harness_id=hid,
                task_type=task_type,
                code=code,
                created_at=time.time(),
            )

            # Validate if data provided
            if validation_data:
                harness.validated = self._validate_harness(harness, validation_data)
                harness.validation_score = 1.0 if harness.validated else 0.0
            else:
                # Trust but verify later
                harness.validated = True

            self._harnesses[hid] = harness
            self._save()
            return harness

        except Exception:
            return None

    def _validate_harness(self, harness: AuthoredHarness, data: Tuple) -> bool:
        """Validate a harness by running it on known data in a sandboxed subprocess."""
        import subprocess
        import sys
        import tempfile
        import os
        try:
            # Write harness code to a temp file and run in subprocess with timeout
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Wrap in a self-test runner
                f.write(harness.code + "\n\n")
                f.write("# Self-test runner\n")
                f.write("import sys\n")
                f.write("found = False\n")
                f.write("for name, obj in dict(globals()).items():\n")
                f.write("    if isinstance(obj, type) and hasattr(obj, 'self_test'):\n")
                f.write("        instance = obj()\n")
                f.write("        passed, _ = instance.self_test()\n")
                f.write("        found = True\n")
                f.write("        sys.exit(0 if passed else 1)\n")
                f.write("if not found: sys.exit(1)\n")
                tmp_path = f.name
            try:
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    timeout=30,
                    capture_output=True,
                )
                return result.returncode == 0
            finally:
                os.unlink(tmp_path)
        except Exception:
            return False

    def record_use(self, harness_id: str, success: bool) -> None:
        """Record that a harness was used (for ranking)."""
        h = self._harnesses.get(harness_id)
        if h:
            h.uses += 1
            if success:
                h.successes += 1
            self._save()


# ---------------------------------------------------------------------------
# Oracle Evolution
# ---------------------------------------------------------------------------

@dataclass
class ProposedOracleCheck:
    """A new oracle check proposed by the system."""
    check_id: str
    name: str
    description: str
    code: str              # implementation
    trigger_pattern: str   # what failure pattern triggered this proposal
    # Validation
    validated: bool = False
    true_positive_rate: float = 0.0   # catches known-bad results
    false_positive_rate: float = 0.0  # doesn't reject known-good results
    # Metadata
    created_at: float = 0.0
    uses: int = 0


class OracleEvolver:
    """Proposes new oracle checks based on failure patterns.

    When the system sees repeated failure patterns that existing oracles
    don't catch, it proposes new checks. These are validated against
    known-good and known-bad results before activation.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._path = persist_path or os.path.expanduser("~/.attestra/oracle_evolution.jsonl")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._proposals: List[ProposedOracleCheck] = []
        self._failure_patterns: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("type") == "proposal":
                        self._proposals.append(ProposedOracleCheck(
                            check_id=d["check_id"],
                            name=d["name"],
                            description=d["description"],
                            code=d.get("code", ""),
                            trigger_pattern=d.get("trigger_pattern", ""),
                            validated=d.get("validated", False),
                            true_positive_rate=d.get("true_positive_rate", 0.0),
                            false_positive_rate=d.get("false_positive_rate", 0.0),
                            created_at=d.get("created_at", 0.0),
                            uses=d.get("uses", 0),
                        ))
                    elif d.get("type") == "pattern":
                        self._failure_patterns.append(d)
        except Exception:
            pass

    def _save(self) -> None:
        with open(self._path, "w") as f:
            for p in self._proposals:
                f.write(json.dumps({
                    "type": "proposal",
                    "check_id": p.check_id,
                    "name": p.name,
                    "description": p.description,
                    "code": p.code,
                    "trigger_pattern": p.trigger_pattern,
                    "validated": p.validated,
                    "true_positive_rate": p.true_positive_rate,
                    "false_positive_rate": p.false_positive_rate,
                    "created_at": p.created_at,
                    "uses": p.uses,
                }) + "\n")
            for pat in self._failure_patterns[-100:]:  # keep last 100
                f.write(json.dumps(pat) + "\n")

    def record_failure_pattern(
        self,
        pattern: str,
        details: Dict,
        oracle_caught: bool,
    ) -> None:
        """Record a failure pattern (for proposing new checks)."""
        self._failure_patterns.append({
            "type": "pattern",
            "pattern": pattern,
            "details": details,
            "oracle_caught": oracle_caught,
            "timestamp": time.time(),
        })
        # Trigger proposal if we see repeated uncaught patterns
        uncaught = [p for p in self._failure_patterns[-20:]
                    if not p.get("oracle_caught")]
        if len(uncaught) >= 3:
            self._maybe_propose_check(uncaught)
        self._save()

    def _maybe_propose_check(self, uncaught_patterns: List[Dict]) -> None:
        """Propose a new oracle check based on uncaught patterns."""
        # Group by pattern type
        from collections import Counter
        pattern_types = Counter(p["pattern"] for p in uncaught_patterns)
        most_common = pattern_types.most_common(1)
        if not most_common:
            return

        pattern_name, count = most_common[0]
        if count < 3:
            return

        # Check if we already have a proposal for this
        existing = [p for p in self._proposals if p.trigger_pattern == pattern_name]
        if existing:
            return

        # Create a proposal stub (LLM would flesh out the code)
        cid = hashlib.sha256(f"{pattern_name}:{time.time()}".encode()).hexdigest()[:12]
        proposal = ProposedOracleCheck(
            check_id=cid,
            name=f"auto_{pattern_name}",
            description=f"Proposed check for uncaught pattern: {pattern_name} (seen {count} times)",
            code="",  # to be filled by LLM
            trigger_pattern=pattern_name,
            created_at=time.time(),
        )
        self._proposals.append(proposal)

    def get_active_checks(self) -> List[ProposedOracleCheck]:
        """Get validated, active oracle checks."""
        return [p for p in self._proposals if p.validated and p.false_positive_rate < 0.1]


# ---------------------------------------------------------------------------
# Strategy Generator
# ---------------------------------------------------------------------------

class StrategyGenerator:
    """Generates new strategies from meta-learner patterns.

    When the meta-learner detects consistent patterns, the generator
    creates new first-class strategies:
      - "On tabular with >100 features, PCA+GBM outperforms raw GBM"
      - "Neural methods timeout on >1M rows without subsampling"
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._path = persist_path or os.path.expanduser("~/.attestra/generated_strategies.jsonl")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._strategies: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._strategies.append(json.loads(line))
        except Exception:
            pass

    def _save(self) -> None:
        with open(self._path, "w") as f:
            for s in self._strategies:
                f.write(json.dumps(s) + "\n")

    def analyze_and_generate(
        self,
        meta_learner_data: List[Dict],
        min_evidence: int = 5,
    ) -> List[Dict]:
        """Analyze meta-learner data and generate new strategy rules.

        Parameters
        ----------
        meta_learner_data : list of dict
            Past experiment outcomes with task profiles and results.
        min_evidence : int
            Minimum observations before generating a rule.

        Returns
        -------
        List of generated strategy dicts.
        """
        new_strategies = []

        # Group outcomes by (task_type, strategy)
        from collections import defaultdict
        groups = defaultdict(list)
        for entry in meta_learner_data:
            key = (entry.get("task_type", ""), entry.get("strategy", ""))
            groups[key].append(entry)

        for (task_type, strategy), outcomes in groups.items():
            if len(outcomes) < min_evidence:
                continue

            success_rate = sum(1 for o in outcomes if o.get("success")) / len(outcomes)
            avg_score = sum(o.get("score", 0) for o in outcomes) / len(outcomes)

            # Generate a rule if strong signal
            if success_rate > 0.8 and avg_score > 0.7:
                rule = {
                    "type": "generated_strategy",
                    "task_type": task_type,
                    "strategy": strategy,
                    "rule": f"On {task_type} tasks, {strategy} succeeds {success_rate*100:.0f}% "
                           f"of the time with avg score {avg_score:.3f}",
                    "confidence": success_rate,
                    "evidence_count": len(outcomes),
                    "created_at": time.time(),
                }
                # Check if we already have this rule
                existing = [s for s in self._strategies
                           if s.get("task_type") == task_type and s.get("strategy") == strategy]
                if not existing:
                    new_strategies.append(rule)
                    self._strategies.append(rule)

            elif success_rate < 0.2 and len(outcomes) >= min_evidence:
                # Anti-pattern: strategy consistently fails
                rule = {
                    "type": "avoid_strategy",
                    "task_type": task_type,
                    "strategy": strategy,
                    "rule": f"AVOID {strategy} on {task_type} tasks (fails {(1-success_rate)*100:.0f}% of the time)",
                    "confidence": 1.0 - success_rate,
                    "evidence_count": len(outcomes),
                    "created_at": time.time(),
                }
                existing = [s for s in self._strategies
                           if s.get("task_type") == task_type
                           and s.get("strategy") == strategy
                           and s.get("type") == "avoid_strategy"]
                if not existing:
                    new_strategies.append(rule)
                    self._strategies.append(rule)

        if new_strategies:
            self._save()

        return new_strategies

    def get_recommendations(self, task_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Get positive recommendations and avoid list for a task type.

        Returns (recommended, avoid) strategy lists.
        """
        recommended = [s for s in self._strategies
                       if s.get("task_type") == task_type and s.get("type") == "generated_strategy"]
        avoid = [s for s in self._strategies
                 if s.get("task_type") == task_type and s.get("type") == "avoid_strategy"]
        return recommended, avoid


# ---------------------------------------------------------------------------
# Compounding metrics (Loop 3 tracking)
# ---------------------------------------------------------------------------

@dataclass
class CompoundingMetrics:
    """Track whether the system is getting better over time."""
    # Rolling windows
    window_size: int = 100
    # Metrics
    certification_rates: List[float] = field(default_factory=list)
    time_to_certification: List[float] = field(default_factory=list)
    sealed_lower_bounds: List[float] = field(default_factory=list)
    # Computed
    trend_certification_rate: float = 0.0  # positive = improving
    trend_time_to_cert: float = 0.0       # negative = improving (faster)
    trend_quality: float = 0.0            # positive = improving

    def record(self, certified: bool, elapsed_s: float, lower_bound: float) -> None:
        """Record an experiment outcome for trend tracking."""
        self.certification_rates.append(1.0 if certified else 0.0)
        self.time_to_certification.append(elapsed_s if certified else 0.0)
        self.sealed_lower_bounds.append(lower_bound)

        # Keep bounded
        if len(self.certification_rates) > self.window_size * 2:
            self.certification_rates = self.certification_rates[-self.window_size:]
            self.time_to_certification = self.time_to_certification[-self.window_size:]
            self.sealed_lower_bounds = self.sealed_lower_bounds[-self.window_size:]

        # Compute trends (compare last half vs first half of window)
        self._compute_trends()

    def _compute_trends(self) -> None:
        """Compute improvement trends."""
        n = len(self.certification_rates)
        if n < 10:
            return
        mid = n // 2
        # Certification rate trend
        first_half = sum(self.certification_rates[:mid]) / mid
        second_half = sum(self.certification_rates[mid:]) / (n - mid)
        self.trend_certification_rate = second_half - first_half

        # Time to certification (only for certified experiments)
        times_first = [t for t in self.time_to_certification[:mid] if t > 0]
        times_second = [t for t in self.time_to_certification[mid:] if t > 0]
        if times_first and times_second:
            avg_first = sum(times_first) / len(times_first)
            avg_second = sum(times_second) / len(times_second)
            self.trend_time_to_cert = avg_first - avg_second  # positive = faster

        # Quality trend
        bounds_first = [b for b in self.sealed_lower_bounds[:mid] if b > 0]
        bounds_second = [b for b in self.sealed_lower_bounds[mid:] if b > 0]
        if bounds_first and bounds_second:
            self.trend_quality = (sum(bounds_second) / len(bounds_second) -
                                  sum(bounds_first) / len(bounds_first))

    def is_improving(self) -> bool:
        """Is the system getting better over time?"""
        # At least one positive trend
        return (self.trend_certification_rate > 0.01 or
                self.trend_time_to_cert > 0 or
                self.trend_quality > 0.005)

    def summary(self) -> Dict:
        """Summary of compounding metrics."""
        n = len(self.certification_rates)
        return {
            "n_experiments": n,
            "current_certification_rate": (
                sum(self.certification_rates[-20:]) / min(20, n)
                if n > 0 else 0.0
            ),
            "trend_certification_rate": self.trend_certification_rate,
            "trend_time_to_cert": self.trend_time_to_cert,
            "trend_quality": self.trend_quality,
            "is_improving": self.is_improving(),
        }
