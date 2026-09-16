"""Durable per-UTC-day spend backstop for the PUBLIC deployment.

Public hosting is password-gated, but spend is allowed *behind* the password (the operator's choice). This
module is the hard ceiling so a shared or leaked password cannot drain the RunPod / Anthropic budget. It
counts the two cost drivers -- GPU fan-out runs and LLM intake calls -- per UTC day, estimates their USD
cost, and BLOCKS once any configured ceiling is reached. When blocked, the caller degrades honestly (GPU ->
CPU, LLM intake -> deterministic inference) and the UI says so; it never silently overspends and never fakes
a result.

Contract:
  * `try_charge(kind)` is the ONLY mutating call. `kind in {"gpu","llm"}`. It atomically (a) rolls the day
    over if the UTC date changed, (b) checks the projected count and USD against the caps, (c) if within
    budget, increments + persists + returns allowed=True, else returns allowed=False WITHOUT incrementing.
    Read-modify-write is guarded by a process lock and an atomic file replace, so concurrent request threads
    cannot double-spend or corrupt the counter.
  * `snapshot()` is read-only (for /healthz and the run response).
  * Only charge when the spend will ACTUALLY happen: the server gates GPU only when a RunPod provider is
    live, and LLM only when an Anthropic key is configured (otherwise both are free local fallbacks).

Caps + unit-cost estimates come from env vars with conservative low defaults; override per deployment. The
USD figures are deliberate over-estimates (a ceiling, not an invoice) -- the point is to bound blast radius,
not to bill precisely.

Env:
  VF_DAILY_USD_CAP          hard daily USD ceiling across GPU + LLM         (default 5.00)
  VF_MAX_GPU_RUNS_PER_DAY   hard daily GPU-run count ceiling                (default 100)
  VF_MAX_LLM_CALLS_PER_DAY  hard daily LLM-intake-call count ceiling        (default 200)
  VF_GPU_USD_PER_RUN        USD charged per GPU fan-out run                 (default 0.05)
  VF_LLM_USD_PER_CALL       USD charged per LLM intake call                 (default 0.02)
  VF_SPEND_STATE            path to the durable counter file   (default <repo>/vf_runs/_spend_state.json)
  VF_SPEND_DISABLED         "1" to disable the cap entirely (local dev)     (default off)
"""
import json
import os
import threading
import time

_LOCK = threading.Lock()


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _cfg():
    return {
        "disabled": os.environ.get("VF_SPEND_DISABLED", "") == "1",
        "usd_cap": _f("VF_DAILY_USD_CAP", 5.0),
        "max_gpu": _i("VF_MAX_GPU_RUNS_PER_DAY", 100),
        "max_llm": _i("VF_MAX_LLM_CALLS_PER_DAY", 200),
        "gpu_usd": _f("VF_GPU_USD_PER_RUN", 0.05),
        "llm_usd": _f("VF_LLM_USD_PER_CALL", 0.02),
    }


def _state_path():
    p = os.environ.get("VF_SPEND_STATE")
    if p:
        return p
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "vf_runs", "_spend_state.json")


def _today():
    # UTC calendar day, e.g. "2026-06-16" -- the budget window
    return time.strftime("%Y-%m-%d", time.gmtime())


def _load():
    path = _state_path()
    try:
        with open(path) as f:
            st = json.load(f)
        if not isinstance(st, dict):
            raise ValueError("not a dict")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        st = {}
    today = _today()
    if st.get("day") != today:                       # day rollover (or first run) -> reset counters
        st = {"day": today, "gpu_runs": 0, "llm_calls": 0, "usd": 0.0}
    # defensive defaults if a field is missing/corrupt
    st.setdefault("day", today)
    st.setdefault("gpu_runs", 0)
    st.setdefault("llm_calls", 0)
    st.setdefault("usd", 0.0)
    return st


def _save(st):
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)            # atomic on POSIX -> a concurrent reader never sees a half-written file


def try_charge(kind):
    """Atomically attempt to charge one unit of `kind` ('gpu' or 'llm') against today's budget.

    Returns (allowed: bool, reason: str|None, snapshot: dict). On allowed=True the counter is incremented and
    persisted; on allowed=False nothing is mutated and `reason` explains which ceiling was hit. Unknown kinds
    are a programming error and raise.
    """
    if kind not in ("gpu", "llm"):
        raise ValueError(f"unknown spend kind {kind!r}")
    cfg = _cfg()
    if cfg["disabled"]:
        return True, None, {"disabled": True}
    unit = cfg["gpu_usd"] if kind == "gpu" else cfg["llm_usd"]
    cnt_key = "gpu_runs" if kind == "gpu" else "llm_calls"
    cnt_cap = cfg["max_gpu"] if kind == "gpu" else cfg["max_llm"]
    with _LOCK:
        st = _load()
        # project this charge and test EVERY ceiling before committing
        if st[cnt_key] + 1 > cnt_cap:
            return False, f"daily {kind} count cap reached ({cnt_cap}/day)", _public(st, cfg)
        if st["usd"] + unit > cfg["usd_cap"] + 1e-9:
            return False, f"daily USD cap reached (${cfg['usd_cap']:.2f}/day)", _public(st, cfg)
        st[cnt_key] += 1
        st["usd"] = round(st["usd"] + unit, 6)
        _save(st)
        return True, None, _public(st, cfg)


def _public(st, cfg):
    return {
        "day": st["day"],
        "gpu_runs": st["gpu_runs"], "max_gpu_runs": cfg["max_gpu"],
        "llm_calls": st["llm_calls"], "max_llm_calls": cfg["max_llm"],
        "usd_spent": round(st["usd"], 4), "usd_cap": cfg["usd_cap"],
        "usd_remaining": round(max(0.0, cfg["usd_cap"] - st["usd"]), 4),
    }


def snapshot():
    """Read-only view of today's spend vs. the caps (no mutation)."""
    cfg = _cfg()
    if cfg["disabled"]:
        return {"disabled": True}
    with _LOCK:
        return _public(_load(), cfg)


# --------------------------------------------------------------------------- self-test
def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    os.environ["VF_SPEND_STATE"] = os.path.join(d, "spend.json")
    os.environ.pop("VF_SPEND_DISABLED", None)
    # tiny caps so we can hit them fast
    os.environ["VF_DAILY_USD_CAP"] = "0.10"
    os.environ["VF_MAX_GPU_RUNS_PER_DAY"] = "3"
    os.environ["VF_MAX_LLM_CALLS_PER_DAY"] = "50"
    os.environ["VF_GPU_USD_PER_RUN"] = "0.02"
    os.environ["VF_LLM_USD_PER_CALL"] = "0.01"
    p = f = 0

    def check(name, cond):
        nonlocal p, f
        if cond:
            print(f"  PASS  {name}"); p += 1
        else:
            print(f"  FAIL  {name}"); f += 1

    # count cap: 3 GPU runs allowed, 4th blocked on the count ceiling (3*0.02=0.06 < 0.10 so USD is not the binder)
    oks = [try_charge("gpu")[0] for _ in range(3)]
    check("first 3 GPU charges allowed", oks == [True, True, True])
    allowed, reason, snap = try_charge("gpu")
    check("4th GPU charge blocked", allowed is False)
    check("block reason names the count cap", reason and "count cap" in reason)
    check("blocked charge did not increment", snap["gpu_runs"] == 3)

    # USD cap binds across kinds: spent 0.06 on GPU, cap 0.10 -> 4 LLM calls (0.04) ok, 5th would hit 0.11 -> blocked
    oks2 = [try_charge("llm")[0] for _ in range(4)]
    check("4 LLM charges within USD headroom allowed", oks2 == [True, True, True, True])
    allowed2, reason2, _ = try_charge("llm")
    check("LLM charge that would exceed USD cap blocked", allowed2 is False and "USD" in (reason2 or ""))

    # durability: a fresh load (simulating a restart) sees the same day's counters
    st = _load()
    check("counters persist across reload", st["gpu_runs"] == 3 and st["llm_calls"] == 4)

    # day rollover resets
    st["day"] = "1999-01-01"
    _save(st)
    snap2 = snapshot()
    check("UTC day rollover resets counters", snap2["gpu_runs"] == 0 and snap2["llm_calls"] == 0)

    # disabled bypasses entirely
    os.environ["VF_SPEND_DISABLED"] = "1"
    check("disabled flag always allows", try_charge("gpu")[0] is True and snapshot().get("disabled") is True)
    os.environ.pop("VF_SPEND_DISABLED", None)

    print(f"  ---- {p} passed, {f} failed ----")
    return f


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)
