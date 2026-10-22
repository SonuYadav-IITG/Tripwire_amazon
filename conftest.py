import os
from datetime import date

import pytest

os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_REGION", "us-east-1")

from tripwire.service import Tripwire  # noqa: E402
from tripwire.store import SqliteStore  # noqa: E402


class Clock:
    def __init__(self):
        self.d = date(2026, 10, 13)

    def __call__(self):
        return self.d


@pytest.fixture(autouse=True)
def offline_ai(monkeypatch):
    monkeypatch.setenv("TRIPWIRE_AI", "offline")


@pytest.fixture(autouse=True)
def fresh_breaker():
    from tripwire import ai
    ai.BREAKER.reset()
    yield
    ai.BREAKER.reset()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store():
    return SqliteStore(":memory:")


@pytest.fixture
def svc(store, clock):
    return Tripwire(store, household="t", today=clock)


INTERNET = ("We're going with the 200 megabit internet plan because it's cheapest. "
            "We'd switch if the price goes above $50 or if outages hit more than twice a month.")
