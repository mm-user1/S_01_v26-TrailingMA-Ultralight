"""Server client lifecycle; application storage isolation is inherited from tests."""

import pytest

from ui.server import app

pytest.register_assert_rewrite(f"{__package__}._helpers")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(app.config, "TESTING", True)
    with app.test_client() as test_client:
        yield test_client
