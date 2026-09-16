"""Pytest fixtures for the tests/ suite.

test_e2e.py is dual-mode: runnable directly (its main() threads a built goal through tests 4/5) and
collectable by pytest. Under pytest, test_resume_idempotence(passed_goal) and
test_predict_from_reload(passed_goal) request this `passed_goal` fixture, which builds the same
achievable-bar goal once and shares it across the session (the e2e build is expensive).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def passed_goal():
    from test_e2e import build_passed_goal, _NO_DATA, _NO_DATA_REASON
    if _NO_DATA:
        pytest.skip(_NO_DATA_REASON)
    return build_passed_goal()
