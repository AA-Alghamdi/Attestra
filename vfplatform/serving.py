"""Servable artifact registry: persist a CERTIFIED winner (estimator + featurizer + label map), versioned,
with predict() and rollback(). This closes the 'vfplatform certifies but does not serve' gap -- one path
now runs goal -> certificate -> deployable artifact -> predict, instead of the certificate living only in
the loop while the legacy lane held the serving.

Discipline: ONLY a certified winner is registered (you deploy only what cleared the bound + latency + cost
gates). Each register() writes an immutable versioned artifact and advances a CURRENT pointer; rollback()
just re-points CURRENT at an earlier version (artifacts are never mutated). predict() refuses to serve a
goal that has no certified version (honest: no silent default).
"""
import json
import os
import pickle
import time

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if _VF not in sys.path:
    sys.path.insert(0, _VF)
from vectorforge import science


def _safe(name):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))[:80]


class NotServable(Exception):
    """Raised when asked to serve a goal/version with no certified artifact."""


class ModelRegistry:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _exp_dir(self, experiment):
        d = os.path.join(self.root, _safe(experiment))
        os.makedirs(d, exist_ok=True)
        return d

    def register(self, experiment, *, goal, estimator, featurizer, label_map, certificate, kind, task_type):
        """Persist a certified winner as a new immutable version and make it CURRENT. Returns the version id.
        estimator/featurizer must be picklable (sklearn estimators + the platform featurizers are)."""
        if not (certificate or {}).get("certified"):
            raise NotServable("refusing to register a non-certified winner (deploy only what is certified)")
        if estimator is None:
            raise NotServable("no local estimator to serve (e.g. a torch winner fit only on a remote worker)")
        d = self._exp_dir(experiment)
        ver = "v" + science.digest({"e": experiment, "c": certificate.get("winner_content_id"),
                                    "t": time.time()}).split(":")[-1][:10]
        vdir = os.path.join(d, ver)
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "model.pkl"), "wb") as fh:
            pickle.dump({"estimator": estimator, "featurizer": featurizer, "label_map": label_map,
                         "kind": kind, "task_type": task_type}, fh)
        meta = {"version": ver, "goal": goal, "kind": kind, "task_type": task_type,
                "certificate": certificate, "created": time.time()}
        with open(os.path.join(vdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        with open(os.path.join(d, "_versions.jsonl"), "a") as fh:                # append-only version log
            fh.write(json.dumps({"version": ver, "created": meta["created"],
                                 "lower_bound": certificate.get("lower_bound"),
                                 "metric": certificate.get("metric")}) + "\n")
        self._set_current(experiment, ver)
        return ver

    def _set_current(self, experiment, version):
        with open(os.path.join(self._exp_dir(experiment), "CURRENT"), "w") as fh:
            fh.write(version)

    def current(self, experiment):
        p = os.path.join(self._exp_dir(experiment), "CURRENT")
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return fh.read().strip()

    def versions(self, experiment):
        d = self._exp_dir(experiment)
        return sorted(v for v in os.listdir(d) if v.startswith("v") and os.path.isdir(os.path.join(d, v)))

    def meta(self, experiment, version=None):
        version = version or self.current(experiment)
        if not version:
            raise NotServable(f"no certified version for {experiment!r}")
        with open(os.path.join(self._exp_dir(experiment), version, "meta.json")) as fh:
            return json.load(fh)

    def rollback(self, experiment, version):
        if version not in self.versions(experiment):
            raise NotServable(f"version {version!r} not found for {experiment!r}")
        self._set_current(experiment, version)
        return version

    def _load(self, experiment, version=None):
        version = version or self.current(experiment)
        if not version:
            raise NotServable(f"no certified/deployed version for {experiment!r}")
        if version not in self.versions(experiment):          # explicit-but-unknown pin -> honest refusal
            raise NotServable(f"version {version!r} not found for {experiment!r}")
        with open(os.path.join(self._exp_dir(experiment), version, "model.pkl"), "rb") as fh:
            return pickle.load(fh), version

    def predict(self, experiment, rows, version=None):
        """Serve predictions from the certified artifact. Featurizes the rows with the SAME fitted
        featurizer and decodes classifier outputs back to original labels. Refuses if nothing certified."""
        art, _ = self._load(experiment, version)
        X = art["featurizer"].transform(list(rows))
        pred = art["estimator"].predict(X)
        if art["task_type"] == "regression" or not art["label_map"]:
            return [float(p) for p in pred]
        inv = {i: lab for lab, i in art["label_map"].items()}
        return [inv.get(int(p), int(p)) for p in pred]
