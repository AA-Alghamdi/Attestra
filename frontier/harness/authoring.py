"""Author-able harness fabric (Phase 3, P3 "author-able harness").

What a *harness* is, and why it must be authored, not hardcoded
---------------------------------------------------------------
The Phase-0 spine knows exactly one task type: tabular classification/regression,
already presented as a `frontier.task.Task` (X, y, kind, theta, metric). To "build
harnesses in the areas needed" the system must, when it meets a NOVEL task type with
no existing adapter, WRITE the adapter that turns that task type's raw data into the
`Task` shape the certify path understands -- and then earn the right to use it.

A harness here is a small, self-contained Python module that defines:

    def build_task(raw) -> dict
        # raw is whatever the task type hands over (e.g. a (texts, labels) tuple,
        # an image-feature matrix, a timeseries window matrix). It returns a plain
        # dict {"X": 2d-float-list, "y": list, "kind": "...", "metric": "...",
        #       "theta": float, "name": "..."} -- i.e. the ingredients of a Task.

That is the entire contract. The harness's job is the data->(X,y,kind,metric) adapter
the ROADMAP names; everything downstream (split, sandbox, sealed certificate) is the
frozen Phase-0 path and is NOT re-implemented here.

The integrity gate (the load-bearing invariant)
------------------------------------------------
ROADMAP Phase 3: "a harness's numbers are trusted only after the harness passes its own
sealed self-test." So an authored harness is NEVER registered on the strength of "the LLM
wrote it." It must first reconstruct a *known-good benchmark* -- a benchmark for that task
type whose answer we already know is achievable -- and the reconstructed Task must CERTIFY
through the unmodified `frontier.engine.ResearchEngine` (true 3-way split, sandboxed
candidate, one-peek sealed Clopper-Pearson / bootstrap bound). Only a certified self-test
admits the harness. Anything else is rejected WITH REASONS. This is exactly the spine's
"certified result or honest decline" discipline applied to the harness itself.

Untrusted code never escapes the sandbox
-----------------------------------------
The LLM-authored `build_task` is itself untrusted code. We do not exec it in-process.
We reuse `frontier.sandbox`'s real subprocess substrate (rlimits + wall-clock timeout +
process-group kill) by shipping a tiny *adapter program* whose `build_estimator()` runs
the authored harness on the known-good raw data and returns the reconstructed (X, y) as
an estimator's predictions, so the harness output is materialized OUT of process. The
materialized Task is then certified by the frozen engine. The numeric firewall is intact:
the authored code produces data only; every promotion-bearing number is computed by
`vectorforge.science` / `vfplatform.sealed` via the engine, never by the harness.

Honest degradation
-------------------
When `llm_client is None` there is no author. We do not fabricate a harness. `author_harness`
returns an `AuthoredHarness` with `status="inactive"` and `accepted=False`; the registry
refuses to register it. This mirrors `LLMProposer` returning [] and the engine reporting
`llm_active=False`.

# === WIRING ===
# The integrator plugs this into the engine layer as a PRE-LOOP capability, not inside the
# per-round proposal loop:
#
#   from frontier.harness.authoring import HarnessRegistry, KnownGoodBenchmark
#   from frontier.engine import EngineConfig, ResearchEngine
#
#   registry = HarnessRegistry()                     # holds only certified harnesses
#   bench    = KnownGoodBenchmark.builtin_text()     # or .builtin_tabular(); a task type
#                                                    # with a known-achievable theta
#   authored = registry.author_and_register(
#       task_type="short-text topic classification",
#       benchmark=bench,
#       llm_client=cfg.llm_client,                   # the SAME Callable[[str],str]|None the
#                                                    # engine already carries on EngineConfig
#       config=EngineConfig(rounds=1, wall_seconds=45, cpu_seconds=40),
#   )
#   if authored.accepted:
#       task = authored.harness.to_task(new_raw_data)   # now trusted: build a Task for the
#       result = ResearchEngine(cfg).run(task)           # NEW data of this task type
#   else:
#       ...                                              # authored.reasons explains the reject;
#                                                        # fall back to a built-in tabular harness
#
# Argument shapes / ordering the integrator must honor:
#   - llm_client(prompt:str) -> str returns ONLY the harness module source (defines build_task).
#   - benchmark.raw is whatever build_task consumes; benchmark.theta is the known-achievable
#     promotion threshold for the self-test; benchmark.kind/metric describe the certified axis.
#   - author_and_register runs the FULL frozen engine once (the self-test) before admitting the
#     harness; it touches the benchmark's sealed split exactly once (engine invariant preserved).
#   - The registry is the trust boundary: only certified harnesses are stored and reused.
#
# Reuses by import (never recomputes a promotion number, never edits Phase 0):
#   frontier.sandbox.run_program  (isolation of authored data-construction code)
#   frontier.task.Task            (the adapter target shape)
#   frontier.engine.ResearchEngine / EngineConfig  (the frozen certify-on-sealed self-test)
#   frontier.certify (transitively, via the engine)  -> vectorforge.science + vfplatform.sealed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .. import sandbox
from ..engine import EngineConfig, EngineResult, ResearchEngine
from ..program import Program
from ..task import Task


# --------------------------------------------------------------------------- known-good benchmark

@dataclass
class KnownGoodBenchmark:
    """A task type's *answer sheet*: raw data plus a threshold we already know is clearable.

    `raw` is whatever the authored harness's build_task() consumes. `kind`/`metric`/`theta`
    describe the certified axis. `theta` is deliberately conservative (well below the achievable
    score for a competent harness) so the self-test cleanly separates a correct adapter from a
    broken one without anchoring to any target paper's number -- it is a floor, not a target.
    """

    name: str
    raw: Any
    kind: str
    theta: float
    metric: str = ""
    description: str = ""

    @staticmethod
    def builtin_tabular() -> "KnownGoodBenchmark":
        """A tabular benchmark (sklearn breast-cancer) presented as the task type's raw form.

        Raw form here is a dict {"data": 2d-list, "target": list} -- the kind of payload a
        generic tabular task type would hand a harness. A correct harness maps it to (X, y).
        """
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        raw = {"data": d.data.tolist(), "target": [str(int(t)) for t in d.target]}
        return KnownGoodBenchmark(
            name="breast_cancer_tabular", raw=raw, kind="classification",
            theta=0.85, metric="accuracy",
            description="binary tabular classification; competent adapter reaches ~0.95.",
        )

    @staticmethod
    def builtin_text() -> "KnownGoodBenchmark":
        """A NON-TABULAR task type: short-text topic classification (raw = (texts, labels)).

        This is the ROADMAP Phase-3 acceptance shape -- one non-tabular task type running end
        to end and certifying through the same gate. The harness must vectorize text into X.
        We use sklearn's 20-newsgroups restricted to a 2-topic, easily-separable pair so a
        correct bag-of-words adapter clears a conservative theta and a broken one (e.g. one that
        emits constant features) cannot.
        """
        from sklearn.datasets import fetch_20newsgroups
        cats = ["rec.sport.hockey", "sci.space"]
        try:
            bunch = fetch_20newsgroups(subset="train", categories=cats,
                                       remove=("headers", "footers", "quotes"),
                                       shuffle=True, random_state=0)
            texts = list(bunch.data)
            labels = [cats[t] for t in bunch.target]
        except Exception:
            # No network / no cached corpus: degrade to a self-contained synthetic text task
            # built from disjoint vocabularies so a real vectorizer separates the classes. This
            # keeps the self-test deterministic and offline-safe (no fabricated certification:
            # the engine still computes the real sealed bound on this synthetic-but-honest data).
            texts, labels = _synthetic_text_corpus()
        return KnownGoodBenchmark(
            name="newsgroups_2topic_text", raw=(texts, labels), kind="classification",
            theta=0.80, metric="accuracy",
            description="short-text 2-topic classification; bag-of-words adapter clears ~0.9+.",
        )


def _synthetic_text_corpus(n_per_class: int = 120, seed: int = 0):
    """Offline fallback corpus: two classes with disjoint vocabularies + shared noise words.

    A correct vectorizing harness recovers the class signal; a degenerate harness cannot.
    """
    rng = np.random.default_rng(seed)
    vocab_a = ["puck", "rink", "goalie", "slapshot", "hockey", "skate"]
    vocab_b = ["orbit", "rocket", "galaxy", "nebula", "telescope", "comet"]
    noise = ["the", "and", "a", "of", "to", "in", "is", "it"]
    texts, labels = [], []
    for cls, vocab in (("hockey", vocab_a), ("space", vocab_b)):
        for _ in range(n_per_class):
            k = int(rng.integers(6, 14))
            words = list(rng.choice(vocab, size=int(rng.integers(3, 7))))
            words += list(rng.choice(noise, size=k))
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(cls)
    return texts, labels


# --------------------------------------------------------------------------- authored harness

@dataclass
class AuthoredHarness:
    """A harness produced by author_harness, possibly not yet trusted.

    `accepted` flips to True only after the self-test certifies. Until then, `to_task` refuses
    to run (no untrusted adapter is used on real data before it earns trust). `reasons` always
    explains the status -- accepted or rejected -- so the decision is auditable.
    """

    task_type: str
    code: str
    status: str                                  # "inactive"|"authored"|"accepted"|"rejected"
    accepted: bool = False
    reasons: List[str] = field(default_factory=list)
    self_test_certificate: Optional[dict] = None
    self_test_summary: str = ""
    provenance: dict = field(default_factory=dict)

    def to_task(self, raw: Any, *, wall_seconds: float = 45.0, cpu_seconds: int = 40) -> Task:
        """Run the (now trusted) authored harness on `raw` and return a frontier Task.

        Even though the harness is accepted, we still materialize its output OUT of process
        (the firewall does not weaken for accepted code: it produces data, never numbers).
        """
        if not self.accepted:
            raise RuntimeError(
                f"harness for {self.task_type!r} is not accepted (status={self.status}); "
                f"reasons: {'; '.join(self.reasons) or 'n/a'}"
            )
        spec = _materialize_task_spec(self.code, raw, wall_seconds=wall_seconds,
                                      cpu_seconds=cpu_seconds)
        if spec is None or "error" in spec:
            raise RuntimeError(
                f"accepted harness failed to build a Task from new data: "
                f"{(spec or {}).get('error', 'no output')}"
            )
        return _task_from_spec(spec)


# --------------------------------------------------------------------------- the prompt + parsing

def _author_prompt(task_type: str, benchmark: KnownGoodBenchmark) -> str:
    """Prompt the LLM to author a harness module for `task_type`.

    The prompt fully specifies the build_task contract and shows the raw payload's shape so the
    author writes a correct adapter. We do NOT reveal the benchmark's labels-as-answers; we
    describe the structure only, so the harness is written to be general, not memorized.
    """
    raw = benchmark.raw
    if isinstance(raw, tuple) and len(raw) == 2:
        shape_hint = (f"raw is a tuple (items, labels): items is a list of {type(raw[0][0]).__name__} "
                      f"(e.g. {raw[0][0]!r:.80}), labels is a list of class names.")
    elif isinstance(raw, dict):
        keys = sorted(raw.keys())
        shape_hint = (f"raw is a dict with keys {keys}; raw['data'] is a list of feature rows "
                      f"(list of floats), raw['target'] is a list of class names.")
    else:
        shape_hint = f"raw is of type {type(raw).__name__}."
    return (
        "You are authoring a DATA HARNESS for a novel ML task type, to plug into a certifier.\n"
        f"Task type: {task_type}\n"
        f"Benchmark kind: {benchmark.kind}; selection/promotion metric: "
        f"{benchmark.metric or 'accuracy'}.\n"
        f"Raw data shape: {shape_hint}\n\n"
        "Write a self-contained Python module that defines exactly one function:\n"
        "    def build_task(raw) -> dict\n"
        "returning {'X': <list of equal-length float rows>, 'y': <list of labels>, "
        "'kind': '<classification|regression>', 'metric': '<metric or empty>', "
        "'theta': <float>, 'name': '<short name>'}.\n"
        "X must be purely numeric (vectorize text, e.g. with sklearn CountVectorizer/TfidfVectorizer "
        "then .toarray()). y must align row-for-row with X. Do NOT fit any model, do NOT print, "
        "do NOT read files or the network, no markdown fences. Imports allowed: numpy, sklearn.\n"
    )


def _strip_fences(code: str) -> str:
    """Remove accidental markdown fences a model may emit despite instructions."""
    c = code.strip()
    if c.startswith("```"):
        lines = c.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        c = "\n".join(lines)
    return c


# --------------------------------------------------------------------------- out-of-process materialize

# The authored harness's build_task() is untrusted. We run it inside frontier.sandbox by wrapping
# it as a Program whose build_estimator() returns a tiny estimator that, when .predict() is called,
# emits a JSON-encoded task spec row-by-row. The sandbox already loads (Xtr, ytr, Xev) arrays and
# calls fit/predict; we ignore those arrays and feed the raw payload through the candidate code.
# The raw payload is injected as a literal (json-serializable) so the child reconstructs it.

_ADAPTER_TEMPLATE = '''\
import json
import numpy as np

# ---- authored harness (untrusted) ----
{authored_code}
# ---- end authored harness ----

_RAW = json.loads({raw_json!r})

class _HarnessAdapter:
    """Runs the authored build_task on the injected raw payload; emits the spec as predictions."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        spec = build_task(_RAW)
        # Validate minimally here so a structurally-broken spec surfaces as a clean error,
        # not a confusing downstream crash. The trusted parent re-validates regardless.
        if not isinstance(spec, dict):
            raise ValueError("build_task must return a dict")
        for k in ("X", "y", "kind"):
            if k not in spec:
                raise KeyError("build_task spec missing key: %s" % k)
        Xout = np.asarray(spec["X"], dtype=float)
        if Xout.ndim != 2:
            raise ValueError("spec X must be 2-D")
        if len(spec["y"]) != Xout.shape[0]:
            raise ValueError("spec X/y length mismatch")
        payload = {{
            "X": Xout.tolist(),
            "y": [str(v) for v in spec["y"]] if spec.get("kind") == "classification"
                 else [float(v) for v in spec["y"]],
            "kind": str(spec["kind"]),
            "metric": str(spec.get("metric", "")),
            "name": str(spec.get("name", "authored_task")),
        }}
        # one prediction element carrying the whole spec as a json string
        return np.asarray([json.dumps(payload)], dtype=object)

def build_estimator():
    return _HarnessAdapter()
'''


def _materialize_task_spec(authored_code: str, raw: Any, *, wall_seconds: float,
                           cpu_seconds: int) -> Optional[dict]:
    """Execute authored build_task(raw) in the sandbox; return the task spec dict (or {'error':..}).

    The numeric firewall is preserved: the child returns DATA (a json spec) only. No metric is
    computed in the child. Resource limits + wall clock + process-group kill all apply.
    """
    try:
        raw_json = json.dumps(raw)
    except (TypeError, ValueError) as e:
        return {"error": f"raw payload is not json-serializable: {e}"}

    code = _ADAPTER_TEMPLATE.format(authored_code=authored_code, raw_json=raw_json)
    prog = Program(code=code, source="harness", label="harness_materialize")

    # The sandbox requires (Xtr, ytr, Xev) arrays; the adapter ignores them, but they must be
    # non-empty and numeric. A 1x1 dummy fits/predicts trivially and triggers exactly one predict.
    dummy_X = np.zeros((2, 1), dtype=float)
    dummy_y = np.array(["a", "b"], dtype=object)
    dummy_eval = np.zeros((1, 1), dtype=float)

    res = sandbox.run_program(prog, dummy_X, dummy_y, dummy_eval, kind="classification",
                              wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
    if not res.ok:
        return {"error": f"[{res.error_kind}] {res.error}"}
    if not res.preds:
        return {"error": "harness produced no output"}
    try:
        spec = json.loads(res.preds[0])
    except Exception as e:
        return {"error": f"harness output was not a valid task spec: {e}"}
    return spec


def _task_from_spec(spec: dict, theta: Optional[float] = None) -> Task:
    """Turn a materialized spec dict into a frontier Task (the trusted parent constructs it)."""
    X = np.asarray(spec["X"], dtype=float)
    kind = spec["kind"]
    if kind == "classification":
        y = np.asarray([str(v) for v in spec["y"]])
    else:
        y = np.asarray([float(v) for v in spec["y"]], dtype=float)
    return Task(X=X, y=y, kind=kind, theta=(theta if theta is not None else 0.0),
                metric=spec.get("metric", ""), name=spec.get("name", "authored_task"))


# --------------------------------------------------------------------------- author + gate

def author_harness(task_type: str, benchmark: KnownGoodBenchmark,
                   llm_client: Optional[Callable[[str], str]] = None,
                   *, config: Optional[EngineConfig] = None,
                   _authored_code: Optional[str] = None) -> AuthoredHarness:
    """Author a harness for `task_type` and run the known-good self-test gate.

    Pipeline:
      1. If no author is available (llm_client is None AND no `_authored_code` stand-in),
         return status="inactive", accepted=False -- honest degradation, no fabrication.
      2. Get harness source: from `_authored_code` (deterministic stand-in, for tests / seeds)
         or from llm_client(prompt). Strip stray fences. Require a build_task definition.
      3. Materialize the benchmark's raw data through the authored harness IN THE SANDBOX
         (untrusted code returns data only).
      4. Build a Task from the materialized spec, with theta = benchmark.theta, and run the
         FROZEN ResearchEngine self-test once. The harness is ACCEPTED iff that run certifies
         on the held-out sealed test. Otherwise REJECTED with the reason.

    `_authored_code` is a labelled shortcut: it lets a caller (or test) supply a fixed harness
    body to exercise the accept/reject gate without an LLM. It is NEVER the promoter -- the
    sealed certificate is. It is the harness analogue of the catalog "seeds": a fallback author.
    """
    cfg = config or EngineConfig(rounds=1, wall_seconds=45, cpu_seconds=40)
    prov = {"task_type": task_type, "benchmark": benchmark.name, "theta": benchmark.theta}

    # ---- step 1: is there an author at all?
    if _authored_code is None and llm_client is None:
        return AuthoredHarness(
            task_type=task_type, code="", status="inactive", accepted=False,
            reasons=["no llm_client and no stand-in author provided; harness authoring inactive "
                     "(honest decline, not fabricated)"],
            provenance=prov,
        )

    # ---- step 2: obtain harness source
    if _authored_code is not None:
        code = _strip_fences(_authored_code)
        prov["author"] = "stand_in"
    else:
        try:
            code = _strip_fences(llm_client(_author_prompt(task_type, benchmark)))
        except Exception as e:
            return AuthoredHarness(task_type=task_type, code="", status="rejected", accepted=False,
                                   reasons=[f"llm_client raised while authoring: {e}"],
                                   provenance=prov)
        prov["author"] = "llm"

    if not code or "def build_task" not in code:
        return AuthoredHarness(task_type=task_type, code=code or "", status="rejected",
                               accepted=False,
                               reasons=["authored module does not define build_task(raw)"],
                               provenance=prov)

    # ---- step 3: materialize the known-good benchmark through the authored harness (sandboxed)
    spec = _materialize_task_spec(code, benchmark.raw, wall_seconds=cfg.wall_seconds,
                                  cpu_seconds=cfg.cpu_seconds)
    if spec is None or "error" in spec:
        return AuthoredHarness(task_type=task_type, code=code, status="rejected", accepted=False,
                               reasons=[f"harness failed to build the benchmark Task: "
                                        f"{(spec or {}).get('error', 'no output')}"],
                               provenance=prov)
    if spec.get("kind") != benchmark.kind:
        return AuthoredHarness(task_type=task_type, code=code, status="rejected", accepted=False,
                               reasons=[f"harness kind {spec.get('kind')!r} != benchmark kind "
                                        f"{benchmark.kind!r}"],
                               provenance=prov)

    # ---- step 4: the gate -- the reconstructed Task must CERTIFY through the frozen engine
    try:
        task = _task_from_spec(spec, theta=benchmark.theta)
        # honor the benchmark's intended metric even if the harness left it blank
        if benchmark.metric:
            task.metric = benchmark.metric
    except Exception as e:
        return AuthoredHarness(task_type=task_type, code=code, status="rejected", accepted=False,
                               reasons=[f"materialized spec is not a valid Task: {e}"],
                               provenance=prov)

    try:
        result: EngineResult = ResearchEngine(cfg).run(task)
    except Exception as e:
        return AuthoredHarness(task_type=task_type, code=code, status="rejected", accepted=False,
                               reasons=[f"self-test engine run raised: {e}"],
                               provenance=prov)

    cert = result.certificate
    if result.certified and cert is not None:
        return AuthoredHarness(
            task_type=task_type, code=code, status="accepted", accepted=True,
            reasons=[f"self-test certified on held-out sealed benchmark {benchmark.name!r}: "
                     f"lower_bound={cert.get('lower_bound')} > theta={cert.get('theta')} "
                     f"(peeks={cert.get('peeks')})"],
            self_test_certificate=cert, self_test_summary=result.summary(), provenance=prov,
        )

    # not certified -> honest reject with the engine's own reason
    reason = result.decline_reason or "self-test did not certify"
    if cert is not None:
        reason += (f" (sealed lower_bound={cert.get('lower_bound')} <= theta={cert.get('theta')})")
    return AuthoredHarness(task_type=task_type, code=code, status="rejected", accepted=False,
                           reasons=[f"self-test on {benchmark.name!r} did not certify: {reason}"],
                           self_test_certificate=cert,
                           self_test_summary=result.summary(), provenance=prov)


# --------------------------------------------------------------------------- registry (trust boundary)

class HarnessRegistry:
    """The trust boundary: holds ONLY harnesses that passed their self-test.

    A harness is the data->Task adapter for a task type. The registry is how the rest of the
    system asks "do we have a trusted harness for this task type?" -- and the only way one gets
    in is through `author_and_register`, which runs the certified self-test gate. There is no
    side door (no `register(code)` that bypasses certification), by design.
    """

    def __init__(self):
        self._harnesses: Dict[str, AuthoredHarness] = {}

    def author_and_register(self, task_type: str, benchmark: KnownGoodBenchmark,
                            llm_client: Optional[Callable[[str], str]] = None,
                            *, config: Optional[EngineConfig] = None,
                            _authored_code: Optional[str] = None) -> AuthoredHarness:
        """Author, gate, and (iff accepted) register a harness for `task_type`.

        Returns the AuthoredHarness either way. Registration happens only on acceptance, so the
        registry can never hand out an uncertified adapter.
        """
        authored = author_harness(task_type, benchmark, llm_client, config=config,
                                   _authored_code=_authored_code)
        if authored.accepted:
            self._harnesses[task_type] = authored
        return authored

    def get(self, task_type: str) -> Optional[AuthoredHarness]:
        """Return the trusted harness for a task type, or None if none is registered."""
        return self._harnesses.get(task_type)

    def has(self, task_type: str) -> bool:
        return task_type in self._harnesses

    def task_types(self) -> List[str]:
        return sorted(self._harnesses.keys())


# --------------------------------------------------------------------------- reference harnesses
# These are deterministic, correct harness bodies usable as the `_authored_code` stand-in when no
# LLM is wired. They are SEEDS/FALLBACK authors (never the promoter -- the sealed certificate is),
# and they double as the known-good reference the test gate exercises.

REFERENCE_TABULAR_HARNESS = '''\
def build_task(raw):
    """Tabular adapter: raw = {'data': rows, 'target': labels} -> (X, y) directly."""
    X = [[float(v) for v in row] for row in raw["data"]]
    y = [str(t) for t in raw["target"]]
    return {"X": X, "y": y, "kind": "classification", "metric": "accuracy",
            "theta": 0.85, "name": "tabular_authored"}
'''

REFERENCE_TEXT_HARNESS = '''\
def build_task(raw):
    """Text adapter: raw = (texts, labels) -> bag-of-words count matrix as X."""
    from sklearn.feature_extraction.text import CountVectorizer
    texts, labels = raw
    vec = CountVectorizer(min_df=1, max_features=400)
    X = vec.fit_transform(texts).toarray()
    return {"X": [[float(v) for v in row] for row in X.tolist()],
            "y": [str(l) for l in labels],
            "kind": "classification", "metric": "accuracy",
            "theta": 0.80, "name": "text_authored"}
'''

# A deliberately BROKEN harness: it MIS-ALIGNS labels to features by permuting the targets, so
# the (X, y) pairs carry no learnable signal. Every estimator still fits and predicts cleanly
# (so this is a genuine PREDICTIVE failure, not a crash), but accuracy collapses to chance and
# the sealed lower bound cannot clear a non-trivial theta. The gate must REJECT this -- proving
# the certifier, not the author's say-so, is the promoter. Used by the test for the reject path.
BROKEN_CONSTANT_HARNESS = '''\
def build_task(raw):
    """Broken adapter: shuffles labels out of correspondence with features -> signal destroyed."""
    import numpy as np
    if isinstance(raw, dict):
        X = [[float(v) for v in row] for row in raw["data"]]
        y = [str(t) for t in raw["target"]]
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        texts, labels = raw
        X = CountVectorizer(min_df=1, max_features=400).fit_transform(texts).toarray()
        X = [[float(v) for v in row] for row in X.tolist()]
        y = [str(l) for l in labels]
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(y))            # break the X<->y correspondence
    y = [y[i] for i in perm]
    return {"X": X, "y": y, "kind": "classification", "metric": "accuracy",
            "theta": 0.80, "name": "broken_misaligned"}
'''
