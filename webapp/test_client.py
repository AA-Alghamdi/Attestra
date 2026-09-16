"""End-to-end test client for the VectorForge web app. stdlib-only HTTP + SSE consumer."""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8765"


def post_run(body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/run", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def consume(run_id):
    """Read the SSE stream; return (events, result). urllib de-chunks the HTTP
    transfer, so the body is clean SSE: messages separated by a blank line, each
    `event: <name>\\ndata: <json>`. Parse by splitting the whole body on blank lines."""
    req = urllib.request.Request(BASE + "/events/" + run_id)
    events, result = [], None
    with urllib.request.urlopen(req) as r:
        body = r.read().decode("utf-8")
    for block in body.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        ev_name, data_lines = None, []
        for line in block.split("\n"):
            if line.startswith(":"):                       # heartbeat comment
                continue
            if line.startswith("event:"):
                ev_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if ev_name is None or not data_lines:
            continue
        payload = json.loads("".join(data_lines))
        if ev_name == "event":
            events.append(payload)
        elif ev_name == "result":
            result = payload
    return events, result


def run_one(body):
    status, j = post_run(body)
    assert status == 200, f"POST /run failed: {status} {j}"
    rid = j["run_id"]
    events, result = consume(rid)
    return rid, events, result


def stages(events):
    return [e["stage"] for e in events]


def assert_subsequence(seq, sub, label):
    """Assert `sub` appears as an ordered (not necessarily contiguous) subsequence of seq."""
    it = iter(seq)
    ok = all(any(x == s for x in it) for s in sub)
    assert ok, f"[{label}] expected ordered {sub} within {seq}"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "1"):
        print("=== TEST 1: breast_cancer, accuracy>=0.92 (expect CERTIFIED) ===")
        rid, ev, res = run_one({"dataset": "breast_cancer", "goal": "classify, accuracy>=0.92",
                                "threshold": 0.92, "metric": "accuracy"})
        st = stages(ev)
        print("run_id:", rid)
        print("stages:", st)
        assert_subsequence(st, ["intake", "split", "audit", "fanout", "val_bound",
                                 "sealed_certify", "done"], "T1")
        assert res["decision"] == "certified", f"expected certified, got {res['decision']}"
        assert res["certified"] is True, "expected certified=True"
        c = res["certificate"]
        print("certificate:", json.dumps({k: c.get(k) for k in
              ("certified", "observed", "lower_bound", "theta", "n", "val_ece",
               "latency_ok", "cost_ok")}))
        print("winner:", res["winner"])
        print("PASS T1\n")

    if which in ("all", "2"):
        print("=== TEST 2a: wine, accuracy>=0.90 (expect CERTIFIED) ===")
        rid, ev, res = run_one({"dataset": "wine", "goal": "classify wine, accuracy>=0.90",
                                "threshold": 0.90, "metric": "accuracy"})
        print("stages:", stages(ev))
        print("decision:", res["decision"], "certified:", res["certified"])
        assert res["certified"] is True, f"wine expected certified, got {res}"
        print("certificate:", json.dumps({k: res["certificate"].get(k) for k in
              ("observed", "lower_bound", "theta", "n")}))
        print("PASS T2a\n")

        print("=== TEST 2b: digits, accuracy>=0.995 (above ceiling -> HONEST-STOP + failure_report) ===")
        rid, ev, res = run_one({"dataset": "digits", "goal": "classify digits, accuracy>=0.995",
                                "threshold": 0.995, "metric": "accuracy"})
        st = stages(ev)
        print("stages:", st)
        term = ev[-1]
        print("terminal event:", term["stage"], term["status"])
        print("decision:", res["decision"], "certified:", res["certified"])
        assert res["certified"] is False, "digits@0.995 should NOT certify (above ceiling)"
        assert res["decision"] in ("honest_stop", "do_not_certify"), f"got {res['decision']}"
        assert term["stage"] in ("honest_stop", "done"), f"unexpected terminal {term['stage']}"
        fr = res["failure_report"]
        assert fr and "dominant_source" in fr, f"missing failure_report: {res}"
        print("failure_report:", json.dumps(fr, indent=2))
        print("PASS T2b\n")

    if which in ("all", "3"):
        print("=== TEST 3: malformed request (unknown dataset) -> 400 ===")
        status, j = post_run({"dataset": "nonexistent_xyz", "goal": "x", "threshold": 0.9})
        print("status:", status, "body:", j)
        assert status == 400, f"expected 400, got {status}"
        assert "unknown dataset" in j.get("error", ""), f"unexpected error: {j}"
        print("--- also: missing goal -> 400 ---")
        status2, j2 = post_run({"dataset": "wine", "threshold": 0.9})
        print("status:", status2, "body:", j2)
        assert status2 == 400 and "goal" in j2.get("error", ""), f"missing-goal not 400: {j2}"
        print("--- also: bad run_id on /events -> 404 ---")
        try:
            urllib.request.urlopen(BASE + "/events/deadbeefdead")
            print("ERROR: expected 404"); sys.exit(1)
        except urllib.error.HTTPError as e:
            print("events bad id status:", e.code)
            assert e.code == 404
        print("PASS T3\n")

    if which in ("all", "4"):
        print("=== TEST 4: two concurrent runs -> streams must not cross ===")
        import threading
        results = {}

        def go(key, body):
            rid, ev, res = run_one(body)
            results[key] = (rid, ev, res)

        ta = threading.Thread(target=go, args=("wine",
              {"dataset": "wine", "goal": "wine acc>=0.90", "threshold": 0.90, "metric": "accuracy"}))
        tb = threading.Thread(target=go, args=("digits",
              {"dataset": "digits", "goal": "digits acc>=0.995", "threshold": 0.995, "metric": "accuracy"}))
        ta.start(); tb.start(); ta.join(); tb.join()
        rid_w, ev_w, res_w = results["wine"]
        rid_d, ev_d, res_d = results["digits"]
        print(f"wine   run_id={rid_w} decision={res_w['decision']} certified={res_w['certified']}")
        print(f"digits run_id={rid_d} decision={res_d['decision']} certified={res_d['certified']}")
        assert rid_w != rid_d, "run_ids collided"
        # each stream's intake event must carry the goal matching its own request -> no crossing
        intake_w = next(e for e in ev_w if e["stage"] == "intake")
        intake_d = next(e for e in ev_d if e["stage"] == "intake")
        assert "wine" in intake_w["goal"] and "digits" in intake_d["goal"], \
            f"streams crossed: wine intake={intake_w['goal']!r} digits intake={intake_d['goal']!r}"
        # and their terminal verdicts must differ as expected (wine certifies, digits does not)
        assert res_w["certified"] is True and res_d["certified"] is False, \
            "concurrent verdicts wrong -> possible cross-talk"
        # n_records sanity: wine=178, digits=1797 — distinct, confirms separate data
        nrw = intake_w.get("n_records"); nrd = intake_d.get("n_records")
        print(f"wine n_records={nrw}  digits n_records={nrd}")
        assert nrw != nrd, "n_records identical -> streams may have crossed"
        print("PASS T4\n")

    if which in ("all", "5"):
        print("=== TEST 5: engine=frontier streams its own stage vocabulary + result ===")
        rid, ev, res = run_one({"dataset": "breast_cancer", "goal": "classify malignant vs benign",
                                "threshold": 0.90, "engine": "frontier"})
        st = stages(ev)
        print("run_id:", rid)
        print("stages:", st)
        # frontier emits engine -> intake -> split -> round -> propose -> measure -> done
        assert_subsequence(st, ["intake", "split", "round", "propose", "measure", "done"], "T5")
        assert res["engine"] == "frontier", f"expected engine=frontier, got {res.get('engine')}"
        assert "certified" in res and "decision" in res, f"frontier result missing keys: {res}"
        # the propose stage must carry the Phase-B intelligence proposals (ensemble/feature-eng)
        prop = next(e for e in ev if e["stage"] == "propose")
        assert prop.get("n_proposals", 0) > 0, f"frontier produced no proposals: {prop}"
        print("propose:", json.dumps({k: prop.get(k) for k in ("n_proposals", "n_intelligence")}))
        print("decision:", res["decision"], "certified:", res["certified"],
              "winner_val_score:", res.get("winner_val_score"))
        print("PASS T5\n")

        print("=== TEST 5b: engine=invalid -> 400 ===")
        status, j = post_run({"dataset": "wine", "goal": "x", "threshold": 0.9, "engine": "bogus"})
        print("status:", status, "body:", j)
        assert status == 400 and "engine" in j.get("error", ""), f"bad engine not rejected: {j}"
        print("PASS T5b\n")

    print("ALL REQUESTED TESTS PASSED")
