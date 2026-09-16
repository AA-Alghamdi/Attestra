"""TextHarness: a REAL non-tabular task type (text classification) through the same gate.

What this proves (the Phase-3 acceptance goal)
-----------------------------------------------
The Phase-0 spine (propose -> sandbox -> score-on-val -> certify-on-sealed) is *modality
agnostic*: only the data->Task adapter changes; the certify path is byte for byte the same
audited-sound one (vectorforge.science + vfplatform.sealed). This harness is that adapter for
text classification, and it SELF-CERTIFIES on a known-good synthetic corpus before its numbers
are trusted -- exactly the trust gate `frontier.harness.base.Harness` defines.

A text corpus is `(documents: list[str], labels: list)`. The harness:
  1. fits a *bounded-vocabulary* TfidfVectorizer (sklearn) and materializes a dense float matrix,
     handing a standard `frontier.task.Task` to the spine. tfidf is UNSUPERVISED (it never sees
     labels), so fitting it before the spine splits leaks no target information; the sealed
     certificate is still computed on held-out LABELS via the one-peek gate (see `adapt`).
  2. exposes a *text-appropriate baseline suite* as seed Programs -- linear models
     (LogisticRegression, LinearSVC) and Naive Bayes (MultinomialNB, ComplementNB), the canonical
     strong bag-of-words baselines. These are SEEDS/FALLBACKS, never the promoter.
  3. provides a self-contained, NO-network known-good corpus (`synth_text_corpus`) generated from
     class-specific multinomial token distributions, so `self_test()` needs no downloaded dataset.

# === WIRING ===
# This is the concrete Harness the Phase-3 router resolves for the "text" task-type key. It is
# registered into the shared REGISTRY at import time (bottom of this file) under both "text" and
# "text_classification", so the integrator does exactly what it does for the tabular harness:
#
#     from frontier.harness import lookup
#     from frontier.harness import text as _t          # ensures registration side-effect
#     h = lookup("text")                                # also reachable via "text_classification"
#     ok, cert = h.self_test()                          # GATE: trust nothing until ok is True
#     assert ok, f"text harness failed self-test: {cert.detail}"
#
#     # raw text in, Task out -- X is the list of document strings (NOT a float matrix yet):
#     task = h.adapt(documents, labels, kind="classification", theta=0.80,
#                    metric="macro_f1", name="my_text_clf")
#     #   adapt() fits the bounded-vocab tfidf and returns a Task whose .X is dense float.
#     result = ResearchEngine(EngineConfig(rounds=2, llm_client=client)).run(task)
#
# Ordering / argument-shape contract the integrator MUST preserve (same as base.Harness):
#   1. self_test() returns True BEFORE any adapt() Task is trusted. base.Harness.self_test()
#      runs the harness's adapt->split->baseline path through the REAL Phase-0 ResearchEngine and
#      seeds it with this harness's baseline_suite() via _StaticSeedProposer, so the text baselines
#      are exercised. The frozen sealed certifier (not this harness) decides certification.
#   2. adapt(X, y, *, kind, theta, name, metric): for TextHarness, X is a Sequence[str] of raw
#      documents and y is the parallel sequence of labels. theta is the caller's standard; the
#      harness never invents a promotion-bearing number.
#   3. metric_for("classification") -> "macro_f1" (multiclass-fair, supported by the frozen
#      certifier). split_protocol -> (0.30, 0.20), the Phase-0 default.
#   4. baseline_suite("classification") -> [Program] (linear + NB); seeds/fallbacks, never promoters.
#
# WHY tfidf at adapt() time and not inside each Program: the spine's numeric firewall runs
# untrusted Program code in a subprocess that receives a float matrix and returns predictions.
# Fitting the trusted, deterministic vectorizer in the harness keeps the firewall contract intact
# (untrusted code never sees raw strings, cannot exfiltrate them) and lets the identical certify
# path score text. Bounded max_features keeps the dense matrix small enough to ship through the
# sandbox's .npz job file; for very large corpora the integrator raises sandbox transport to
# sparse -- a substrate swap, not a contract change.
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

# Repo root on sys.path so sibling frontier modules + the sound certifier resolve from anywhere
# (mirrors base.py / certify.py; this file lives under frontier/harness/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.program import Program            # noqa: E402
from frontier.task import Task                  # noqa: E402

from .base import Harness, register             # noqa: E402


# Metrics the frozen certifier supports for classification; we expose only right-axis choices so
# a text task cannot be routed to, say, a regression metric. macro_f1 is the default because text
# corpora are often multiclass / imbalanced and macro_f1 weights classes equally.
_VALID_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
_DEFAULT_METRIC = "macro_f1"


# --------------------------------------------------------------------------- baseline suite
# Each baseline is a Program: code defining build_estimator() that operates on the tfidf FLOAT
# matrix adapt() produced. These mirror the canonical strong bag-of-words baselines and are
# SEEDS/FALLBACKS that guarantee a floor; the frozen certifier is the only promoter.
#
# WHY these four: linear models (logreg / linear SVM) and the two multinomial-event-model Naive
# Bayes variants are the textbook strong baselines for sparse high-dimensional tfidf features.
# ComplementNB is specifically robust to class imbalance vs MultinomialNB. NB requires
# non-negative inputs; tfidf is non-negative by construction, so it is well posed here.
_TEXT_BASELINES: List[Tuple[str, str]] = [
    ("text_logreg",
     "from sklearn.linear_model import LogisticRegression\n"
     "def build_estimator():\n"
     "    return LogisticRegression(C=4.0, max_iter=2000, class_weight='balanced')\n"),
    ("text_linsvc",
     "from sklearn.svm import LinearSVC\n"
     "def build_estimator():\n"
     "    return LinearSVC(C=1.0, class_weight='balanced')\n"),
    ("text_multinomial_nb",
     "from sklearn.naive_bayes import MultinomialNB\n"
     "def build_estimator():\n"
     "    return MultinomialNB(alpha=0.3)\n"),
    ("text_complement_nb",
     "from sklearn.naive_bayes import ComplementNB\n"
     "def build_estimator():\n"
     "    return ComplementNB(alpha=0.3)\n"),
]


def text_baseline_programs() -> List[Program]:
    """The text baseline suite as seed Programs (linear + Naive Bayes). Fresh list per call."""
    return [
        Program(code=code, source="seed", label=name,
                provenance={"suite": "text_baselines", "harness": "text"})
        for name, code in _TEXT_BASELINES
    ]


# --------------------------------------------------------------------------- synthetic corpus

def synth_text_corpus(
    *,
    n_per_class: int = 120,
    n_classes: int = 3,
    vocab_size: int = 60,
    signal_tokens: int = 6,
    doc_len: Tuple[int, int] = (12, 28),
    noise_frac: float = 0.45,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Generate a self-contained, separable text corpus from class-specific token distributions.

    No network. Each class owns a disjoint block of `signal_tokens` high-probability "signal"
    words; the rest of every document is sampled from a shared "noise" vocabulary. With the
    defaults the classes are linearly separable in tfidf space but not trivially so (the
    `noise_frac` shared tokens force the model to actually weight the signal words). That is exactly
    what a *known-good* self-test corpus should be: a competent harness must certify, and a broken
    one (wrong metric axis, label scramble, empty vocab) must fail.

    Returns (documents, labels): documents are space-joined token strings; labels are string class
    names "c0".."c{n_classes-1}". The corpus is shuffled so class blocks are not contiguous.
    """
    if n_classes < 2:
        raise ValueError("need at least 2 classes")
    if signal_tokens * n_classes > vocab_size:
        raise ValueError("signal_tokens * n_classes must be <= vocab_size")
    rng = np.random.default_rng(seed)
    vocab = [f"w{i:03d}" for i in range(vocab_size)]
    signal_idx = {c: list(range(c * signal_tokens, (c + 1) * signal_tokens))
                  for c in range(n_classes)}
    noise_idx = list(range(n_classes * signal_tokens, vocab_size))
    if not noise_idx:
        raise ValueError("no noise tokens left; lower signal_tokens or raise vocab_size")

    docs: List[str] = []
    labels: List[str] = []
    lo, hi = doc_len
    for c in range(n_classes):
        sig = signal_idx[c]
        for _ in range(n_per_class):
            length = int(rng.integers(lo, hi + 1))
            n_noise = int(round(noise_frac * length))
            n_sig = max(1, length - n_noise)
            toks = [vocab[rng.choice(sig)] for _ in range(n_sig)]
            toks += [vocab[rng.choice(noise_idx)] for _ in range(n_noise)]
            rng.shuffle(toks)
            docs.append(" ".join(toks))
            labels.append(f"c{c}")
    order = rng.permutation(len(docs))
    docs = [docs[i] for i in order]
    labels = [labels[i] for i in order]
    return docs, labels


# --------------------------------------------------------------------------- the harness

class TextHarness(Harness):
    """Harness for bag-of-words text classification.

    `adapt()` takes raw document strings (NOT a float matrix), fits a bounded-vocabulary
    TfidfVectorizer, and returns a standard `frontier.task.Task` with a dense float `X` that the
    Phase-0 engine certifies unchanged. `self_test()` (inherited from base.Harness) certifies the
    harness on a known-good synthetic corpus before its numbers are trusted.

    Constructor knobs control the tfidf vocabulary:
      - max_features : hard cap on vocabulary size (keeps the dense matrix shippable through the
                       sandbox .npz). Generous default for real corpora.
      - ngram_max    : include unigrams..ngram_max-grams (1 = unigrams, 2 = + bigrams).
      - min_df       : drop tokens in fewer than this many documents (noise control).
      - sublinear_tf : 1 + log(tf) weighting (standard for text; dampens frequent tokens).
    """

    key = "text"
    kinds = ("classification",)

    def __init__(self, *, max_features: int = 20000, ngram_max: int = 1,
                 min_df: int = 1, sublinear_tf: bool = True):
        super().__init__()
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

    # ------------------------------------------------------------------ adapter API
    def metric_for(self, kind: str) -> str:
        """Right-axis default metric for text classification (macro_f1)."""
        if kind != "classification":
            raise ValueError(f"TextHarness handles 'classification', not {kind!r}")
        return _DEFAULT_METRIC

    def _vectorize(self, documents: Sequence[str]) -> np.ndarray:
        """Fit a bounded-vocab TfidfVectorizer and return a dense float matrix.

        tfidf is an UNSUPERVISED transform (it never sees labels), so fitting it on the full corpus
        before the spine splits does not leak target information. It can see token statistics of
        documents that later land in the sealed split; that is standard transductive feature
        extraction, and is why a production deployment would refit the vectorizer inside the train
        fold. The Phase-0 sealed certificate is still computed on held-out LABELS via the one-peek
        gate, which is the property under test.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, max(1, self.ngram_max)),
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
        )
        Xs = vec.fit_transform(list(documents))
        # Dense float matrix: the Task contract is a float ndarray and the sandbox ships floats.
        return np.asarray(Xs.todense(), dtype=float)

    def adapt(self, X, y, *, kind: str, theta: float, name: str = "task",
              metric: str = "") -> Task:
        """Vectorize raw documents into a Phase-0 Task the engine certifies unchanged.

        `X` is a Sequence[str] of raw documents (NOT a pre-built matrix); `y` is the parallel
        sequence of labels. Validates early rather than emit a silently-corrupt Task:
          - kind must be classification;
          - documents/labels lengths match and the corpus is non-empty;
          - at least 2 distinct labels (classification is ill-posed otherwise);
          - the metric, if supplied, is on the right axis;
          - tfidf produced a non-empty vocabulary.
        theta is the caller's verification standard; the harness does not invent it.
        """
        if kind != "classification":
            raise ValueError(f"TextHarness handles 'classification', not {kind!r}")
        documents = list(X)
        labels = [str(v) for v in y]
        if len(documents) != len(labels):
            raise ValueError(f"documents ({len(documents)}) and labels ({len(labels)}) mismatch")
        if len(documents) == 0:
            raise ValueError("empty corpus")
        if len(set(labels)) < 2:
            raise ValueError("text classification needs >= 2 distinct labels")
        if metric:
            if metric not in _VALID_METRICS:
                raise ValueError(f"metric {metric!r} not valid for classification; "
                                 f"choose one of {_VALID_METRICS}")
        else:
            metric = self.metric_for(kind)

        Xmat = self._vectorize(documents)
        if Xmat.shape[1] == 0:
            raise ValueError("tfidf produced an empty vocabulary (raise max_features / lower min_df)")
        return Task(X=Xmat, y=np.asarray(labels), kind="classification",
                    theta=float(theta), metric=metric, name=name)

    def baseline_suite(self, kind: str) -> List[Program]:
        """Text floor recipes (linear + Naive Bayes), as seed Programs (never promoters)."""
        if kind != "classification":
            raise ValueError(f"TextHarness handles 'classification', not {kind!r}")
        return text_baseline_programs()

    def split_protocol(self, kind: str) -> Tuple[float, float]:
        """Text uses the Phase-0 default 30/20 split (train 50% / val 20% / sealed 30%)."""
        return (0.30, 0.20)

    # ------------------------------------------------------------------ self-test case
    def _self_test_case(self) -> Tuple[Sequence[str], np.ndarray, str, str, float]:
        """Known-good case: a self-contained separable synthetic corpus (NO network).

        3 classes x 120 docs from disjoint signal-token blocks plus shared noise. A competent
        linear/NB baseline reaches high macro_f1; the sealed LOWER bound at this n clears 0.70 with
        margin. We pick 0.70 (well below the achievable score) on purpose: the self-test verifies
        the adapter/metric/split PLUMBING with margin to spare, not the modeling difficulty. The X
        returned is the raw document list -- adapt() vectorizes it, exactly as for a real task.
        """
        docs, labels = synth_text_corpus(seed=0)
        return docs, np.asarray(labels), "classification", "synthetic_separable_text", 0.70


# --------------------------------------------------------------------------- registration

# Register a default instance into the shared HarnessRegistry under the text task-type keys, so a
# router resolving "text" (or the problem-typer alias "text_classification") gets this harness.
# We use a tuned default (bounded vocab + bigrams) suitable for real corpora; a caller wanting
# different tfidf knobs constructs its own TextHarness and registers it under a fresh key.
_TEXT = TextHarness(max_features=20000, ngram_max=2, min_df=1)
try:
    register(_TEXT, "text", "text_classification")
except KeyError:
    # Idempotent under re-import (e.g. pytest reimport): keep the already-registered instance.
    pass
