from __future__ import annotations

import pytest

from app.utils.inmemory_db import FakeDB


@pytest.fixture()
def fake_db() -> FakeDB:
    return FakeDB()


@pytest.fixture
def anyio_backend():
    return "asyncio"

