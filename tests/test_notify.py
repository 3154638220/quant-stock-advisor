"""Tests for src/notify/__init__.py - webhook notification handlers."""

from __future__ import annotations

import json
from unittest import mock

import pytest
import requests

from src.notify import WecomWebhookHandler


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {"errcode": 0, "errmsg": "ok"}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


# ── Initialization ─────────────────────────────────────────────────────────


def test_handler_init_defaults():
    h = WecomWebhookHandler(url="https://example.com/webhook")
    assert h.url == "https://example.com/webhook"
    assert h.timeout == 10.0
    assert h.mention_all is False


def test_handler_init_custom():
    h = WecomWebhookHandler(url="https://example.com/webhook", timeout=5.0, mention_all=True)
    assert h.timeout == 5.0
    assert h.mention_all is True


# ── send_markdown ──────────────────────────────────────────────────────────


@mock.patch("src.notify.requests.post")
def test_send_markdown_success(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 0, "errmsg": "ok"})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_markdown("Hello **world**")
    assert result is True
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["msgtype"] == "markdown"


@mock.patch("src.notify.requests.post")
def test_send_markdown_with_mention_all(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 0, "errmsg": "ok"})
    h = WecomWebhookHandler(url="https://example.com/webhook", mention_all=True)
    result = h.send_markdown("Alert!")
    assert result is True
    payload = mock_post.call_args[1]["json"]
    assert payload["markdown"]["mentioned_list"] == ["@all"]


@mock.patch("src.notify.requests.post")
def test_send_markdown_errcode_nonzero(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 40001, "errmsg": "invalid key"})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_markdown("test")
    assert result is False


@mock.patch("src.notify.requests.post")
def test_send_markdown_http_error(mock_post):
    mock_post.return_value = FakeResponse(500, {})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_markdown("test")
    assert result is False


@mock.patch("src.notify.requests.post")
def test_send_markdown_request_exception(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("timeout")
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_markdown("test")
    assert result is False


@mock.patch("src.notify.requests.post")
def test_send_markdown_json_decode_error(mock_post):
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.side_effect = json.JSONDecodeError("msg", "", 0)
    mock_post.return_value = mock_resp
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_markdown("test")
    assert result is False


# ── send_text ──────────────────────────────────────────────────────────────


@mock.patch("src.notify.requests.post")
def test_send_text_success(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 0, "errmsg": "ok"})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_text("Plain text message")
    assert result is True
    payload = mock_post.call_args[1]["json"]
    assert payload["msgtype"] == "text"


@mock.patch("src.notify.requests.post")
def test_send_text_with_mentions(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 0, "errmsg": "ok"})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_text("Hello @user1", mentioned_list=["user1", "user2"])
    assert result is True
    payload = mock_post.call_args[1]["json"]
    assert payload["text"]["mentioned_list"] == ["user1", "user2"]


@mock.patch("src.notify.requests.post")
def test_send_text_errcode_nonzero(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 40001, "errmsg": "invalid key"})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_text("test")
    assert result is False


@mock.patch("src.notify.requests.post")
def test_send_text_http_error(mock_post):
    mock_post.return_value = FakeResponse(500, {})
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_text("test")
    assert result is False


@mock.patch("src.notify.requests.post")
def test_send_text_request_exception(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("timeout")
    h = WecomWebhookHandler(url="https://example.com/webhook")
    result = h.send_text("test")
    assert result is False


# ── __call__ integration ──────────────────────────────────────────────────


@mock.patch("src.notify.requests.post")
def test_call_sends_markdown(mock_post):
    mock_post.return_value = FakeResponse(200, {"errcode": 0, "errmsg": "ok"})
    h = WecomWebhookHandler(url="https://example.com/webhook")

    class FakeAlert:
        def __str__(self):
            return "IC Decay Alert: score=0.025"

    result = h(FakeAlert())
    assert result is True
    payload = mock_post.call_args[1]["json"]
    assert payload["msgtype"] == "markdown"
    assert "IC Decay" in payload["markdown"]["content"]
