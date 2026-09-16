"""Durable goal store. Each goal lives in its own directory under the store root with state.json, its
raw data, and model artifacts. Goals persist and resume after interruption.
"""

import json
import time
from pathlib import Path

import joblib

from .domain import Goal

ROOT = Path(__file__).resolve().parent.parent / "store"


def _dir(goal_id):
    d = ROOT / goal_id
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d


def save(goal: Goal):
    goal.updated_at = _now()
    d = _dir(goal.id)
    (d / "state.json").write_text(json.dumps(goal.to_json(), indent=2, default=str))
    return goal


def load(goal_id) -> Goal:
    p = ROOT / goal_id / "state.json"
    if not p.exists():
        raise FileNotFoundError(f"goal {goal_id} not found")
    return Goal.from_json(json.loads(p.read_text()))


def exists(goal_id):
    return (ROOT / goal_id / "state.json").exists()


def list_goals():
    out = []
    if not ROOT.exists():
        return out
    for d in sorted(ROOT.iterdir()):
        p = d / "state.json"
        if p.exists():
            g = json.loads(p.read_text())
            out.append({"id": g["id"], "name": g["name"], "status": g["status"],
                        "metric": g["verification"]["metric"], "threshold": g["verification"]["threshold"],
                        "updated_at": g.get("updated_at", "")})
    return out


def write_rows(goal_id, name, rows):
    p = _dir(goal_id) / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def save_artifact(goal_id, name, obj):
    p = _dir(goal_id) / "artifacts" / f"{name}.joblib"
    joblib.dump(obj, p)
    return str(p)


def load_artifact(path):
    return joblib.load(path)


def append_log(goal: Goal, entry: dict):
    entry = {"t": _now(), **entry}
    goal.evidence_log.append(entry)
    return entry


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
