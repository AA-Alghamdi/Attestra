"""Provider-resilience tests (soft edge: no certifier / sealed / theta change). ALL MOCKED -- requests is
monkeypatched, so NO real RunPod call is ever made and NO spend can occur. Covers:
  - classify_transient: 401/403/404/400 persistent; 429/5xx/Timeout/ConnectionError transient.
  - submit_one: retries a TRANSIENT failure then succeeds; FAILS FAST on a 401 (assert call counts).
  - probe(): False+reason on 401 and on all-unhealthy (unhealthy>0, ready=0); True on healthy (ready>0)
             AND True on cold (ready=0, unhealthy=0).
Plain asserts + a main() runner, matching tests/test_runpod.py."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.providers import RunPodProvider, classify_transient, ResourceGated


def run(tests):
    p = f = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return p, f


# ----------------------------------------------------------------- a minimal requests test double
class _FakeResp:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err


class _FakeRequests:
    """Stand-in for the `requests` module. Records calls; serves canned responses/exceptions via callbacks."""
    def __init__(self, post=None, get=None):
        import requests as _real
        self.exceptions = _real.exceptions   # reuse real exception classes for isinstance checks
        self._post, self._get = post, get
        self.post_calls, self.get_calls = [], []

    def post(self, url, **kw):
        self.post_calls.append((url, kw))
        return self._post(url, **kw)

    def get(self, url, **kw):
        self.get_calls.append((url, kw))
        return self._get(url, **kw)


def _patch_requests(monkeypatch_module, fake):
    """Install `fake` as the module returned by `import requests` inside providers.py functions."""
    sys.modules["requests"] = fake


class _Restore:
    """Context manager: swap sys.modules['requests'] for a fake, restore the real module after."""
    def __init__(self, fake):
        self.fake = fake
        self._real = None

    def __enter__(self):
        import requests as _real
        self._real = _real
        sys.modules["requests"] = self.fake
        # also neutralize time.sleep so backoff does not slow the test
        import time
        self._sleep = time.sleep
        time.sleep = lambda *_a, **_k: None
        return self.fake

    def __exit__(self, *exc):
        sys.modules["requests"] = self._real
        import time
        time.sleep = self._sleep
        return False


# --------------------------------------------------------------------------- classify_transient
def test_classify_persistent_client_errors():
    for code in (400, 401, 403, 404):
        assert classify_transient(code) is False, f"HTTP {code} must be persistent (fail fast)"


def test_classify_transient_status_codes():
    assert classify_transient(429) is True, "429 rate-limit is transient"
    for code in (500, 502, 503, 504):
        assert classify_transient(code) is True, f"HTTP {code} (5xx) is transient"


def test_classify_transient_exceptions():
    import requests
    assert classify_transient(requests.exceptions.Timeout("slow")) is True
    assert classify_transient(requests.exceptions.ConnectionError("refused")) is True
    # an HTTPError carrying a 503 is transient; one carrying a 401 is not
    e503 = requests.exceptions.HTTPError("boom"); e503.response = _FakeResp(503)
    e401 = requests.exceptions.HTTPError("nope"); e401.response = _FakeResp(401)
    assert classify_transient(e503) is True
    assert classify_transient(e401) is False
    # plain stdlib timeout / connection errors are transient too
    assert classify_transient(TimeoutError()) is True
    assert classify_transient(ConnectionError()) is True
    # a generic ValueError is persistent (fail fast)
    assert classify_transient(ValueError("bad")) is False


# --------------------------------------------------------------------------- submit_one resilience
def test_submit_one_retries_transient_then_succeeds():
    p = RunPodProvider(api_key="rpa_dummy", endpoint_id="ep_dummy", timeout=60)
    state = {"posts": 0}

    def fake_post(url, **kw):
        state["posts"] += 1
        if state["posts"] == 1:
            # first /run: transient connection blip -> should retry
            import requests
            raise requests.exceptions.ConnectionError("transient blip")
        return _FakeResp(200, {"id": "job123"})

    def fake_get(url, **kw):
        return _FakeResp(200, {"status": "COMPLETED", "output": {"y_pred_val": [1, 0]}})

    fake = _FakeRequests(post=fake_post, get=fake_get)
    with _Restore(fake):
        out = p.submit_one({"op": "fit_val"})
    assert out == {"y_pred_val": [1, 0]}, out
    assert state["posts"] == 2, f"expected 1 retry (2 /run posts), got {state['posts']}"


def test_submit_one_fails_fast_on_401():
    p = RunPodProvider(api_key="rpa_dummy", endpoint_id="ep_dummy", timeout=60)
    state = {"posts": 0}

    def fake_post(url, **kw):
        state["posts"] += 1
        return _FakeResp(401, {"error": "unauthorized"}, text="unauthorized")

    def fake_get(url, **kw):
        raise AssertionError("must not poll after a fail-fast /run")

    fake = _FakeRequests(post=fake_post, get=fake_get)
    raised = False
    with _Restore(fake):
        try:
            p.submit_one({"op": "fit_val"})
        except Exception as e:  # noqa: BLE001
            raised = True
            assert not isinstance(e, ResourceGated)  # it's an HTTP fail, not a config gate
    assert raised, "401 must raise"
    assert state["posts"] == 1, f"401 must fail fast (1 /run post, no retry), got {state['posts']}"


def test_submit_one_retries_5xx_then_fails_after_cap():
    # a persistent 5xx (server keeps 500ing) is transient PER-attempt but should stop at max_retries.
    p = RunPodProvider(api_key="rpa_dummy", endpoint_id="ep_dummy", timeout=60)
    state = {"posts": 0}

    def fake_post(url, **kw):
        state["posts"] += 1
        return _FakeResp(503, {}, text="server error")

    def fake_get(url, **kw):
        raise AssertionError("never reached")

    fake = _FakeRequests(post=fake_post, get=fake_get)
    raised = False
    with _Restore(fake):
        try:
            p.submit_one({"op": "fit_val"}, max_retries=3)
        except Exception:  # noqa: BLE001
            raised = True
    assert raised, "exhausted 5xx retries must raise"
    # 1 initial + 3 retries = 4 attempts
    assert state["posts"] == 4, f"expected 4 attempts (1 + 3 retries), got {state['posts']}"


# --------------------------------------------------------------------------- probe()
def _provider_with_health(workers, http=200):
    p = RunPodProvider(api_key="rpa_dummy", endpoint_id="ep_dummy")
    p.health_summary = lambda: ({"http": http, "workers": workers, "jobs": {}} if http == 200
                                else {"http": http, "error": f"HTTP {http}"})
    return p


def test_probe_not_configured():
    p = RunPodProvider(api_key=None, endpoint_id=None)
    r = p.probe()
    assert r["available"] is False and r["reason"] == "RunPod not configured", r


def test_probe_false_on_401():
    p = _provider_with_health({}, http=401)
    r = p.probe()
    assert r["available"] is False and "auth rejected" in r["reason"], r


def test_probe_false_on_unreachable():
    p = _provider_with_health({}, http=500)
    r = p.probe()
    assert r["available"] is False and "unreachable (HTTP 500)" in r["reason"], r


def test_probe_false_on_all_unhealthy():
    p = _provider_with_health({"unhealthy": 2, "ready": 0, "idle": 0, "initializing": 0, "running": 0})
    r = p.probe()
    assert r["available"] is False and "workers unhealthy" in r["reason"], r


def test_probe_true_on_healthy():
    p = _provider_with_health({"unhealthy": 0, "ready": 2, "idle": 0, "initializing": 0, "running": 1})
    r = p.probe()
    assert r["available"] is True and r["reason"] is None, r


def test_probe_true_on_cold():
    # cold serverless: 0 ready, 0 unhealthy -> cold start will spin a worker; must NOT be blocked.
    p = _provider_with_health({"unhealthy": 0, "ready": 0, "idle": 0, "initializing": 0, "running": 0})
    r = p.probe()
    assert r["available"] is True and r["reason"] is None, r


def test_probe_true_when_some_unhealthy_but_some_ready():
    # mixed: 1 unhealthy but 1 ready -> still usable, do not block.
    p = _provider_with_health({"unhealthy": 1, "ready": 1, "idle": 0, "initializing": 0, "running": 0})
    r = p.probe()
    assert r["available"] is True and r["reason"] is None, r


def test_health_summary_never_raises_on_network_error():
    p = RunPodProvider(api_key="rpa_dummy", endpoint_id="ep_dummy")

    def fake_get(url, **kw):
        import requests
        raise requests.exceptions.ConnectionError("dns fail")

    fake = _FakeRequests(post=None, get=fake_get)
    with _Restore(fake):
        h = p.health_summary()
    assert h["http"] == 0 and "error" in h, h


def test_health_summary_ok_shape():
    p = RunPodProvider(api_key="rpa_dummy", endpoint_id="ep_dummy")

    def fake_get(url, **kw):
        return _FakeResp(200, {"workers": {"ready": 1}, "jobs": {"inQueue": 0}})

    fake = _FakeRequests(post=None, get=fake_get)
    with _Restore(fake):
        h = p.health_summary()
    assert h["http"] == 200 and h["workers"] == {"ready": 1} and h["jobs"] == {"inQueue": 0}, h


TESTS = [test_classify_persistent_client_errors, test_classify_transient_status_codes,
         test_classify_transient_exceptions, test_submit_one_retries_transient_then_succeeds,
         test_submit_one_fails_fast_on_401, test_submit_one_retries_5xx_then_fails_after_cap,
         test_probe_not_configured, test_probe_false_on_401, test_probe_false_on_unreachable,
         test_probe_false_on_all_unhealthy, test_probe_true_on_healthy, test_probe_true_on_cold,
         test_probe_true_when_some_unhealthy_but_some_ready,
         test_health_summary_never_raises_on_network_error, test_health_summary_ok_shape]

if __name__ == "__main__":
    print("== test_resilience (provider soft edge: resilience + honest probe) ==")
    p, f = run(TESTS)
    raise SystemExit(1 if f else 0)
