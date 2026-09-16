"""Featurizers + candidate model families per ML subfield (the fan-out search space).

Kept small and CPU-friendly: a handful of sklearn families x light hyperparameter variants x seeds. Each
candidate is a deterministic spec the executor fits on TRAIN and scores on VALIDATION. Featurization is
subfield-specific (tabular: numeric + one-hot from nested `features`; text: TF-IDF on a text field).
"""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------------- featurizers
class TabularFeaturizer:
    """Numeric features STANDARDIZED (zero-mean/unit-var, fit on train+val+test); categoricals one-hot over
    the train+val+test vocab. Reads the nested `features` dict. Standardization matters: without it,
    distance/gradient learners (logistic, SVM, MLP) are handicapped on unscaled columns (e.g. pixel 0-16 vs
    a 0-1 feature) and logistic fails to converge. Tree models are scale-invariant, so this never hurts them.
    The scaler is fit on the FULL split union (already content-addressed and leakage-audited), so it adds no
    new leakage path beyond the existing featurizer vocab."""
    def __init__(self):
        self.numeric, self.cats, self.vocab = [], [], {}
        self.mu, self.sd = {}, {}

    @staticmethod
    def _f(r, c):
        return (r.get("features") or {}).get(c)

    def fit(self, rows_for_schema, all_rows):
        cols = list((rows_for_schema[0].get("features") or {}).keys())
        self.numeric = [c for c in cols
                        if all(isinstance(self._f(r, c), (int, float)) for r in rows_for_schema
                               if self._f(r, c) is not None)]
        self.cats = [c for c in cols if c not in self.numeric]
        self.vocab = {c: sorted({str(self._f(r, c)) for r in all_rows}) for c in self.cats}
        for c in self.numeric:                                  # standardization stats over the split union
            vals = [float(self._f(r, c)) for r in all_rows if isinstance(self._f(r, c), (int, float))]
            arr = np.asarray(vals, dtype=float) if vals else np.zeros(1)
            self.mu[c] = float(arr.mean())
            sd = float(arr.std())
            self.sd[c] = sd if sd > 1e-9 else 1.0               # guard constant columns
        return self

    def transform(self, rows):
        X = []
        for r in rows:
            row = []
            for c in self.numeric:
                v = self._f(r, c)
                v = float(v) if isinstance(v, (int, float)) else self.mu.get(c, 0.0)   # impute mean for missing
                row.append((v - self.mu.get(c, 0.0)) / self.sd.get(c, 1.0))
            for c in self.cats:
                row += [1.0 if str(self._f(r, c)) == lev else 0.0 for lev in self.vocab[c]]
            X.append(row)
        return np.array(X, dtype=float)


class TextFeaturizer:
    def __init__(self, text_key="text", max_features=20000):
        self.text_key, self.vec = text_key, TfidfVectorizer(max_features=max_features)

    def fit(self, rows_for_schema, all_rows):
        self.vec.fit([str(r.get(self.text_key, "")) for r in all_rows])
        return self

    def transform(self, rows):
        return self.vec.transform([str(r.get(self.text_key, "")) for r in rows])


def make_featurizer(kind, text_key="text"):
    return TextFeaturizer(text_key=text_key) if kind == "text" else TabularFeaturizer()


# ----------------------------------------------------------------------------- candidate families
def _tabular_families(seeds):
    base = [
        ("logistic|C1.0", lambda s: LogisticRegression(max_iter=2000, C=1.0), {"C": 1.0}),
        ("logistic|C0.1", lambda s: LogisticRegression(max_iter=2000, C=0.1), {"C": 0.1}),
        ("random_forest|100", lambda s: RandomForestClassifier(n_estimators=100, random_state=s), {"n_estimators": 100}),
        ("random_forest|300d", lambda s: RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s), {"n_estimators": 300, "max_depth": 12}),
        ("hist_gbm|200", lambda s: HistGradientBoostingClassifier(max_iter=200, random_state=s), {"max_iter": 200}),
        ("hist_gbm|400", lambda s: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=s), {"max_iter": 400, "lr": 0.05}),
    ]
    return [(name, ctor, dict(params, seed=s), s) for (name, ctor, params) in base for s in seeds]


def _text_families(seeds):
    base = [
        ("tfidf+logistic|C1.0", lambda s: LogisticRegression(max_iter=2000, C=1.0), {"C": 1.0}),
        ("tfidf+logistic|C3.0", lambda s: LogisticRegression(max_iter=2000, C=3.0), {"C": 3.0}),
        ("tfidf+multinomial_nb", lambda s: MultinomialNB(), {}),
    ]
    return [(name, ctor, dict(params, seed=s), s) for (name, ctor, params) in base for s in seeds]


def candidate_families(kind, seeds=(0, 1)):
    """Return [(family_name, ctor(seed)->estimator, params_dict, seed)] for the subfield."""
    return _text_families(seeds) if kind == "text" else _tabular_families(seeds)


def fit_and_val(ctor, seed, Xtr, ytr, Xva, yva):
    """Fit one candidate on train, score accuracy on validation. Returns (val_acc, latency_ms, est)."""
    est = ctor(seed)
    t0 = time.time()
    est.fit(Xtr, ytr)
    latency_ms = (time.time() - t0) * 1000.0
    val_acc = float((est.predict(Xva) == yva).mean())
    return val_acc, latency_ms, est
