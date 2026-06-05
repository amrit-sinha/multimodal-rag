import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.core.security as security


def test_api_key_disabled_allows_any(monkeypatch):
    monkeypatch.setattr(security.settings, "api_key", "")
    # Should not raise regardless of header value.
    asyncio.run(security.require_api_key(x_api_key=None))
    asyncio.run(security.require_api_key(x_api_key="anything"))


def test_api_key_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(security.settings, "api_key", "secret")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(security.require_api_key(x_api_key="wrong"))
    assert exc.value.status_code == 401


def test_api_key_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(security.settings, "api_key", "secret")
    asyncio.run(security.require_api_key(x_api_key="secret"))


def test_rate_limit_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(security.settings, "rate_limit_per_minute", 0)
    request = SimpleNamespace(client=SimpleNamespace(host="1.2.3.4"))
    asyncio.run(security.rate_limit(request))
