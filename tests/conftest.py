"""Shared test setup: put the project root on the import path, and pin the clock."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinikit.agent.clinic import TIMEZONE, seeded_db  # noqa: E402

# Monday 14 September 2026, 10:00 Beirut.
# Every time-dependent test uses this. Without a pinned clock, a test that passes at
# 09:00 fails at 11:00, and a test that fails intermittently is worse than no test.
FROZEN_NOW = datetime(2026, 9, 14, 10, 0, tzinfo=TIMEZONE)


@pytest.fixture
def now():
    return FROZEN_NOW


@pytest.fixture
def db():
    """A fresh appointment book for each test, so tests cannot affect each other."""
    return seeded_db("p_001")
