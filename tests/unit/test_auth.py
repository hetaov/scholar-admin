"""付费白名单鉴权（services/auth.py）单元测试

覆盖：
- get_request_openid：dev / enforce 两种模式下 openid 头解析
- is_whitelisted：白名单判断与安全默认（集合/文档缺失一律视为未授权）
- require_paid_user：dev 放行无身份请求；enforce 强制校验；白名单命中/未命中
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import APIRouter, Depends, HTTPException
from starlette.requests import Request

from services.auth import (
    OPENID_HEADER,
    WHITELIST_DOC_ID,
    get_request_openid,
    is_whitelisted,
    require_paid_user,
)
from tests.fakes.fake_db import FakeDB

WHITELIST_COLLECTION = "app_whitelist"


def _run(coro):
    """同步执行 async 代码，避免依赖 pytest-asyncio。"""
    return asyncio.run(coro)


def _whitelist_db(openids: list[str] | None = None) -> FakeDB:
    db = FakeDB()
    if openids is not None:
        db.add(WHITELIST_COLLECTION, {"_id": WHITELIST_DOC_ID, "openids": openids})
    return db


def _make_request(openid: str | None = None) -> Request:
    headers = {}
    if openid is not None:
        headers[OPENID_HEADER] = openid
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/eval/translate",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


# ---------------- get_request_openid ----------------


class TestGetRequestOpenid:
    def test_dev_mode_without_header_returns_empty(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "dev")
        assert get_request_openid(_make_request()) == ""

    def test_enforce_mode_without_header_returns_empty(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        assert get_request_openid(_make_request()) == ""

    def test_with_header_returns_openid(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        assert get_request_openid(_make_request("o-owner")) == "o-owner"

    def test_header_name_case_insensitive(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-wx-openid", b"o-owner")],
        }
        assert get_request_openid(Request(scope)) == "o-owner"


# ---------------- is_whitelisted ----------------


class TestIsWhitelisted:
    def test_empty_openid_rejected(self):
        assert not _run(is_whitelisted("", _whitelist_db(["o-owner"])))

    def test_missing_collection_rejected(self):
        assert not _run(is_whitelisted("o-owner", FakeDB()))

    def test_missing_doc_rejected(self):
        db = FakeDB()
        db.add("other_collection", {"_id": "x"})
        assert not _run(is_whitelisted("o-owner", db))

    def test_openid_not_in_list_rejected(self):
        assert not _run(is_whitelisted("o-stranger", _whitelist_db(["o-owner"])))

    def test_openid_in_list_allowed(self):
        assert _run(is_whitelisted("o-owner", _whitelist_db(["o-owner"])))

    def test_doc_without_openids_rejected(self):
        db = _whitelist_db(None)
        db.add(WHITELIST_COLLECTION, {"_id": WHITELIST_DOC_ID})
        assert not _run(is_whitelisted("o-owner", db))


# ---------------- require_paid_user ----------------


class TestRequirePaidUser:
    def test_dev_mode_no_header_allowed(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "dev")
        result = _run(require_paid_user(_make_request(), _whitelist_db(["o-owner"])))
        assert result == ""

    def test_dev_mode_whitelisted_allowed(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "dev")
        result = _run(
            require_paid_user(_make_request("o-owner"), _whitelist_db(["o-owner"]))
        )
        assert result == "o-owner"

    def test_dev_mode_not_whitelisted_rejected(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "dev")
        with pytest.raises(HTTPException) as exc:
            _run(require_paid_user(_make_request("o-stranger"), _whitelist_db(["o-owner"])))
        assert exc.value.status_code == 403

    def test_enforce_mode_no_header_rejected(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        with pytest.raises(HTTPException) as exc:
            _run(require_paid_user(_make_request(), _whitelist_db(["o-owner"])))
        assert exc.value.status_code == 403

    def test_enforce_mode_whitelisted_allowed(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        result = _run(
            require_paid_user(_make_request("o-owner"), _whitelist_db(["o-owner"]))
        )
        assert result == "o-owner"

    def test_enforce_mode_not_whitelisted_rejected(self, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        with pytest.raises(HTTPException) as exc:
            _run(require_paid_user(_make_request("o-stranger"), _whitelist_db(["o-owner"])))
        assert exc.value.status_code == 403


# ---------------- HTTP 链路（依赖注入 + 403 契约） ----------------


class TestHttpFlow:
    def test_paid_router_returns_403_for_unauthorized(self, make_client, fake_db, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        router = APIRouter()

        @router.get("/paid/ping", dependencies=[Depends(require_paid_user)])
        async def ping():
            return {"ok": True}

        client = make_client(router)
        resp = client.get("/paid/ping")
        assert resp.status_code == 403

    def test_paid_router_allows_whitelisted_openid(self, make_client, fake_db, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        fake_db.add(WHITELIST_COLLECTION, {"_id": WHITELIST_DOC_ID, "openids": ["o-owner"]})
        router = APIRouter()

        @router.get("/paid/ping", dependencies=[Depends(require_paid_user)])
        async def ping():
            return {"ok": True}

        client = make_client(router)
        resp = client.get("/paid/ping", headers={OPENID_HEADER: "o-owner"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_paid_router_blocks_non_whitelisted_openid(self, make_client, fake_db, monkeypatch):
        monkeypatch.setattr("services.auth.AUTH_MODE", "enforce")
        fake_db.add(WHITELIST_COLLECTION, {"_id": WHITELIST_DOC_ID, "openids": ["o-owner"]})
        router = APIRouter()

        @router.get("/paid/ping", dependencies=[Depends(require_paid_user)])
        async def ping():
            return {"ok": True}

        client = make_client(router)
        resp = client.get("/paid/ping", headers={OPENID_HEADER: "o-stranger"})
        assert resp.status_code == 403
