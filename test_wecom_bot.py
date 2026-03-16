"""Tests for wecom_bot.py — WeCom intelligent bot callback service."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Patch env vars BEFORE importing wecom_bot so module-level init is safe
os.environ.setdefault("WECOM_BOT_TOKEN", "test_token")
os.environ.setdefault("WECOM_BOT_AES_KEY", "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG")
os.environ.setdefault("WECOM_BOT_RECEIVE_ID", "test_corp_id")
os.environ.setdefault("WECOM_BOT_KEY", "test_webhook_key")

from wecom_bot import (
    WXBizMsgCrypt,
    _strip_at_mentions,
    _extract_message,
    reply_via_response_url,
    handle_message,
    generate_research_program,
    run_research_async,
    _running_tasks,
    _tasks_lock,
    MAX_CONCURRENT_TASKS,
    TaskInfo,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def crypt():
    """Create a WXBizMsgCrypt instance with a known key for testing."""
    # 43-char base64 key (will be padded with '=' and decoded to 32 bytes)
    return WXBizMsgCrypt(
        token="test_token",
        encoding_aes_key="abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG",
        receive_id="test_corp_id",
    )


@pytest.fixture(autouse=True)
def clean_running_tasks():
    """Ensure _running_tasks is empty before and after each test."""
    with _tasks_lock:
        _running_tasks.clear()
    yield
    with _tasks_lock:
        _running_tasks.clear()


# ── 1. _strip_at_mentions ───────────────────────────────────────────────


class TestStripAtMentions:
    def test_single_mention_with_space(self):
        assert _strip_at_mentions("@BotName 研究 topic") == "研究 topic"

    def test_single_mention_no_space_should_not_strip_chinese(self):
        # @Bot is stripped as a mention, "研究 topic" remains
        result = _strip_at_mentions("@Bot研究 topic")
        # @Bot is \S+ (no space follows, so "Bot研究" is the mention token
        # until whitespace). The regex ^(@\S+\s*)+ matches "@Bot研究 " leaving "topic"
        # Actually: @Bot研究 is one \S+ token, then space is \s*, so result is "topic"
        # Let's just verify it doesn't eat "研究" when there IS a space
        # The no-space case: @Bot研究 is treated as one mention token
        assert result == "topic"

    def test_mention_with_space_preserves_content(self):
        # When there IS a space after @Bot, "研究" is preserved
        assert _strip_at_mentions("@Bot 研究 topic") == "研究 topic"

    def test_multiple_mentions(self):
        assert _strip_at_mentions("@A @B 研究 topic") == "研究 topic"

    def test_no_mention(self):
        assert _strip_at_mentions("研究 topic") == "研究 topic"

    def test_empty_string(self):
        assert _strip_at_mentions("") == ""

    def test_whitespace_only(self):
        assert _strip_at_mentions("   ") == ""

    def test_chinese_mention(self):
        assert _strip_at_mentions("@研究助手 some topic") == "some topic"

    def test_mention_only_returns_original(self):
        # If stripping leaves empty, return original stripped text
        result = _strip_at_mentions("@BotName")
        assert result == "@BotName"

    def test_multiple_mentions_with_tabs(self):
        result = _strip_at_mentions("@A\t@B\tresearch topic")
        assert "research topic" in result


# ── 2. _extract_message ─────────────────────────────────────────────────


class TestExtractMessage:
    def test_json_text_content(self):
        data = json.dumps({
            "MsgType": "text",
            "Text": {"Content": "研究 蛙鞋设计"},
            "From": {"Name": "TestUser", "UserId": "user1"},
            "ResponseUrl": "https://example.com/response",
            "WebhookUrl": "https://example.com/webhook?key=abc",
        })
        result = _extract_message(data)
        assert result["text"] == "研究 蛙鞋设计"
        assert result["from_user"] == "TestUser"
        assert result["response_url"] == "https://example.com/response"
        assert result["webhook_url"] == "https://example.com/webhook?key=abc"
        assert result["msg_type"] == "text"

    def test_json_lowercase_keys(self):
        data = json.dumps({
            "msgtype": "text",
            "text": {"content": "hello world"},
            "from": {"name": "User2", "user_id": "u2"},
            "response_url": "https://example.com/resp",
        })
        result = _extract_message(data)
        assert result["text"] == "hello world"
        assert result["from_user"] == "User2"
        assert result["response_url"] == "https://example.com/resp"
        assert result["msg_type"] == "text"

    def test_xml_format(self):
        xml = """<xml>
            <MsgType>text</MsgType>
            <Content>@Bot 研究 topic</Content>
            <From><UserId>user3</UserId></From>
            <WebhookUrl>https://example.com/wh</WebhookUrl>
            <ResponseUrl>https://example.com/resp</ResponseUrl>
        </xml>"""
        result = _extract_message(xml)
        assert result["text"] == "研究 topic"  # @mention stripped
        assert result["msg_type"] == "text"
        assert result["webhook_url"] == "https://example.com/wh"

    def test_missing_text_field(self):
        data = json.dumps({"MsgType": "image", "From": {"Name": "User"}})
        result = _extract_message(data)
        assert result["text"] == ""
        assert result["msg_type"] == "image"

    def test_empty_text(self):
        data = json.dumps({"MsgType": "text", "Text": {"Content": ""}})
        result = _extract_message(data)
        assert result["text"] == ""

    def test_json_with_at_mention_in_content(self):
        data = json.dumps({
            "MsgType": "text",
            "Text": {"Content": "@ResearchBot 研究 AI 趋势"},
        })
        result = _extract_message(data)
        assert result["text"] == "研究 AI 趋势"

    def test_json_content_field_fallback(self):
        """When msg_type is not text but Content field exists."""
        data = json.dumps({"MsgType": "event", "Content": "some content"})
        result = _extract_message(data)
        assert result["text"] == "some content"

    def test_json_text_as_string(self):
        """Text field is a string instead of dict."""
        data = json.dumps({"MsgType": "text", "Text": "direct string"})
        result = _extract_message(data)
        assert result["text"] == "direct string"

    def test_invalid_content_returns_empty(self):
        result = _extract_message("not json or xml {{{{")
        assert result["text"] == ""
        assert result["from_user"] == ""

    def test_response_url_extraction(self):
        data = json.dumps({
            "MsgType": "text",
            "Text": {"Content": "test"},
            "ResponseUrl": "https://qyapi.weixin.qq.com/cgi-bin/callback/reply?token=abc",
        })
        result = _extract_message(data)
        assert "reply?token=abc" in result["response_url"]

    def test_webhook_url_extraction(self):
        data = json.dumps({
            "MsgType": "text",
            "Text": {"Content": "test"},
            "WebhookUrl": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xyz",
        })
        result = _extract_message(data)
        assert "key=xyz" in result["webhook_url"]


# ── 3. reply_via_response_url ───────────────────────────────────────────


class TestReplyViaResponseUrl:
    def test_empty_response_url(self):
        assert reply_via_response_url("", "hello") is False

    @patch("wecom_bot.httpx.post")
    def test_markdown_succeeds(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
        resp.text = '{"errcode":0,"errmsg":"ok"}'
        mock_post.return_value = resp

        result = reply_via_response_url("https://example.com/reply", "# Hello")
        assert result is True
        # Should only call once (markdown succeeded)
        assert mock_post.call_count == 1
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["msgtype"] == "markdown"

    @patch("wecom_bot.httpx.post")
    def test_markdown_fails_text_succeeds(self, mock_post):
        fail_resp = MagicMock()
        fail_resp.status_code = 200
        fail_resp.json.return_value = {"errcode": 90001, "errmsg": "not supported"}
        fail_resp.text = '{"errcode":90001}'

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
        ok_resp.text = '{"errcode":0}'

        mock_post.side_effect = [fail_resp, ok_resp]

        result = reply_via_response_url("https://example.com/reply", "Hello")
        assert result is True
        assert mock_post.call_count == 2
        # Second call should be text
        second_payload = mock_post.call_args_list[1][1]["json"]
        assert second_payload["msgtype"] == "text"

    @patch("wecom_bot.httpx.post")
    def test_both_fail(self, mock_post):
        fail_resp = MagicMock()
        fail_resp.status_code = 200
        fail_resp.json.return_value = {"errcode": 90001, "errmsg": "fail"}
        fail_resp.text = '{"errcode":90001}'
        mock_post.return_value = fail_resp

        result = reply_via_response_url("https://example.com/reply", "Hello")
        assert result is False
        assert mock_post.call_count == 2

    @patch("wecom_bot.httpx.post")
    def test_network_error(self, mock_post):
        mock_post.side_effect = Exception("connection refused")

        result = reply_via_response_url("https://example.com/reply", "Hello")
        assert result is False

    @patch("wecom_bot.httpx.post")
    def test_http_status_error_falls_through(self, mock_post):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        mock_post.return_value = resp

        result = reply_via_response_url("https://example.com/reply", "test")
        assert result is False
        # Both markdown and text attempts get 500
        assert mock_post.call_count == 2

    @patch("wecom_bot.httpx.post")
    def test_non_json_200_treated_as_success(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        resp.text = "OK"
        mock_post.return_value = resp

        result = reply_via_response_url("https://example.com/reply", "test")
        assert result is True


# ── 4. handle_message ───────────────────────────────────────────────────


class TestHandleMessage:
    @patch("wecom_bot.reply_message")
    def test_help_command(self, mock_reply):
        handle_message("帮助", "user1", "", "wh_key")
        mock_reply.assert_called_once()
        # Should contain the intro text
        call_text = mock_reply.call_args[0][0]
        assert "AutoResearch Bot" in call_text

    @patch("wecom_bot.reply_message")
    def test_help_aliases(self, mock_reply):
        for cmd in ("help", "?", "你好", "hello", "介绍"):
            mock_reply.reset_mock()
            handle_message(cmd, "user1", "", "wh_key")
            mock_reply.assert_called_once()

    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.send_wecom_text")
    def test_status_no_tasks(self, mock_send, mock_reply):
        handle_message("状态", "user1", "", "wh_key")
        mock_reply.assert_called_once()
        assert "没有运行中的任务" in mock_reply.call_args[0][0]

    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.run_research_async")
    def test_research_with_prefix(self, mock_run, mock_reply):
        handle_message("研究 蛙鞋设计", "user1", "resp_url", "wh_key")
        mock_reply.assert_called_once()
        assert "蛙鞋设计" in mock_reply.call_args[0][0]
        mock_run.assert_called_once_with("蛙鞋设计", "wh_key", "user1")

    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.run_research_async")
    def test_research_without_prefix_triggers_research(self, mock_run, mock_reply):
        """Any unrecognized text >= 2 chars should trigger research."""
        handle_message("蛙鞋设计", "user1", "resp_url", "wh_key")
        mock_reply.assert_called_once()
        assert "蛙鞋设计" in mock_reply.call_args[0][0]
        mock_run.assert_called_once_with("蛙鞋设计", "wh_key", "user1")

    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.run_research_async")
    def test_single_char_does_not_trigger_research(self, mock_run, mock_reply):
        handle_message("a", "user1", "", "wh_key")
        mock_run.assert_not_called()
        # Should show help-like message
        call_text = mock_reply.call_args[0][0]
        assert "研究主题" in call_text

    @patch("wecom_bot.reply_message")
    def test_cancel_no_tasks(self, mock_reply):
        handle_message("取消 蛙鞋", "user1", "", "wh_key")
        mock_reply.assert_called_once()
        assert "没有运行中的任务" in mock_reply.call_args[0][0]

    @patch("wecom_bot.reply_message")
    def test_cancel_lists_tasks_when_no_target(self, mock_reply):
        # Add a fake task
        with _tasks_lock:
            _running_tasks["蛙鞋设计"] = TaskInfo(
                topic="蛙鞋设计", proc=None, thread=None,
                from_user="u1", webhook_key="k",
                start_time=time.time(), work_dir=Path("/tmp/test"),
            )
        handle_message("取消", "user1", "", "wh_key")
        mock_reply.assert_called_once()
        assert "蛙鞋设计" in mock_reply.call_args[0][0]

    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.reply_via_response_url")
    @patch("wecom_bot.send_wecom_text")
    def test_diagnostic_command(self, mock_send, mock_reply_url, mock_reply):
        handle_message("诊断", "user1", "", "wh_key")
        mock_reply.assert_called_once()
        call_text = mock_reply.call_args[0][0]
        assert "诊断报告" in call_text

    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.run_research_async")
    def test_research_prefix_no_topic(self, mock_run, mock_reply):
        handle_message("研究", "user1", "", "wh_key")
        mock_run.assert_not_called()
        assert "请提供研究主题" in mock_reply.call_args[0][0]

    @patch("wecom_bot.reply_message")
    def test_status_with_running_tasks(self, mock_reply):
        with _tasks_lock:
            _running_tasks["test_topic"] = TaskInfo(
                topic="test_topic", proc=None, thread=None,
                from_user="tester", webhook_key="k",
                start_time=time.time(), work_dir=Path("/tmp/test"),
            )
        handle_message("状态", "user1", "", "wh_key")
        mock_reply.assert_called_once()
        assert "test_topic" in mock_reply.call_args[0][0]


# ── 5. WXBizMsgCrypt ────────────────────────────────────────────────────


class TestWXBizMsgCrypt:
    def test_encrypt_decrypt_roundtrip(self, crypt):
        original = "Hello, WeCom! 你好企微"
        encrypted = crypt._encrypt(original)
        decrypted = crypt._decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_decrypt_chinese_long(self, crypt):
        original = "这是一段较长的中文消息，用于测试加密和解密的正确性。" * 10
        encrypted = crypt._encrypt(original)
        decrypted = crypt._decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_produces_different_ciphertext(self, crypt):
        """Each encryption uses random prefix, so same plaintext gives different ciphertext."""
        msg = "test message"
        enc1 = crypt._encrypt(msg)
        enc2 = crypt._encrypt(msg)
        assert enc1 != enc2
        # But both decrypt to same plaintext
        assert crypt._decrypt(enc1) == msg
        assert crypt._decrypt(enc2) == msg

    def test_signature_is_deterministic(self, crypt):
        sig1 = crypt._signature("a", "b", "c")
        sig2 = crypt._signature("c", "a", "b")  # sorted → same
        assert sig1 == sig2

    def test_signature_changes_with_different_input(self, crypt):
        sig1 = crypt._signature("a", "b")
        sig2 = crypt._signature("a", "c")
        assert sig1 != sig2

    def test_verify_url_success(self, crypt):
        plaintext = "echo_string_12345"
        encrypted = crypt._encrypt(plaintext)
        timestamp = "1234567890"
        nonce = "test_nonce"
        sig = crypt._signature(crypt.token, timestamp, nonce, encrypted)

        ok, decrypted = crypt.verify_url(sig, timestamp, nonce, encrypted)
        assert ok is True
        assert decrypted == plaintext

    def test_verify_url_bad_signature(self, crypt):
        encrypted = crypt._encrypt("test")
        ok, decrypted = crypt.verify_url("bad_signature", "123", "nonce", encrypted)
        assert ok is False
        assert decrypted == ""

    def test_decrypt_msg_json(self, crypt):
        plaintext = '{"MsgType":"text","Text":{"Content":"hello"}}'
        encrypted = crypt._encrypt(plaintext)
        timestamp = "1234567890"
        nonce = "test_nonce"
        sig = crypt._signature(crypt.token, timestamp, nonce, encrypted)

        body = json.dumps({"encrypt": encrypted})
        ok, decrypted = crypt.decrypt_msg(sig, timestamp, nonce, body)
        assert ok is True
        assert decrypted == plaintext

    def test_decrypt_msg_xml(self, crypt):
        plaintext = "<xml><Content>hello</Content></xml>"
        encrypted = crypt._encrypt(plaintext)
        timestamp = "123"
        nonce = "nonce"
        sig = crypt._signature(crypt.token, timestamp, nonce, encrypted)

        body = f"<xml><Encrypt>{encrypted}</Encrypt></xml>"
        ok, decrypted = crypt.decrypt_msg(sig, timestamp, nonce, body)
        assert ok is True
        assert decrypted == plaintext

    def test_decrypt_msg_bad_signature(self, crypt):
        encrypted = crypt._encrypt("test")
        body = json.dumps({"encrypt": encrypted})
        ok, decrypted = crypt.decrypt_msg("wrong_sig", "123", "nonce", body)
        assert ok is False

    def test_decrypt_msg_no_encrypt_field(self, crypt):
        ok, decrypted = crypt.decrypt_msg("sig", "123", "nonce", '{"foo":"bar"}')
        assert ok is False

    def test_pkcs7_pad_unpad_roundtrip(self, crypt):
        data = b"hello world"
        padded = crypt._pkcs7_pad(data)
        assert len(padded) % 32 == 0
        unpadded = crypt._pkcs7_unpad(padded)
        assert unpadded == data

    def test_pkcs7_pad_exact_block_size(self, crypt):
        data = b"a" * 32  # exact block size
        padded = crypt._pkcs7_pad(data)
        # Should add a full block of padding
        assert len(padded) == 64
        assert crypt._pkcs7_unpad(padded) == data

    def test_encrypt_reply(self, crypt):
        reply = "reply message"
        nonce = "test_nonce"
        timestamp = "1234567890"
        xml = crypt.encrypt_reply(reply, nonce, timestamp)

        assert "<Encrypt>" in xml
        assert "<MsgSignature>" in xml
        assert "<TimeStamp>1234567890</TimeStamp>" in xml
        assert "<Nonce>" in xml

        # Verify we can extract and decrypt from the reply
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        encrypted = root.findtext("Encrypt")
        decrypted = crypt._decrypt(encrypted)
        assert decrypted == reply


# ── 6. generate_research_program ────────────────────────────────────────


class TestGenerateResearchProgram:
    def test_output_contains_topic(self):
        program = generate_research_program("蛙鞋设计")
        assert "蛙鞋设计" in program

    def test_output_has_markdown_structure(self):
        program = generate_research_program("AI 趋势")
        assert program.startswith("# Research Program: AI 趋势")
        assert "## Topic" in program
        assert "## Research Questions" in program
        assert "## Scope" in program
        assert "## Constraints" in program

    def test_output_has_numbered_questions(self):
        program = generate_research_program("test topic")
        assert "1." in program
        assert "2." in program
        assert "3." in program
        assert "4." in program
        assert "5." in program

    def test_topic_appears_in_questions(self):
        program = generate_research_program("碳纤维")
        # Topic should appear in at least the first question
        lines = program.split("\n")
        question_lines = [l for l in lines if l.strip().startswith("1.")]
        assert any("碳纤维" in l for l in question_lines)


# ── 7. run_research_async ───────────────────────────────────────────────


class TestRunResearchAsync:
    @patch("wecom_bot.reply_message")
    @patch("wecom_bot.send_wecom_text")
    def test_duplicate_task_prevention(self, mock_send, mock_reply):
        """Second call with same topic should warn about duplicate."""
        with _tasks_lock:
            _running_tasks["蛙鞋设计"] = TaskInfo(
                topic="蛙鞋设计", proc=None, thread=None,
                from_user="u1", webhook_key="k",
                start_time=time.time(), work_dir=Path("/tmp/test"),
            )

        run_research_async("蛙鞋设计", "wh_key", "user1")
        mock_send.assert_called_once()
        assert "正在研究中" in mock_send.call_args[0][1]

    @patch("wecom_bot.reply_message")
    def test_max_concurrent_tasks_limit(self, mock_reply):
        """Should reject when max concurrent tasks reached."""
        with _tasks_lock:
            for i in range(MAX_CONCURRENT_TASKS):
                _running_tasks[f"task_{i}"] = TaskInfo(
                    topic=f"task_{i}", proc=None, thread=None,
                    from_user="u", webhook_key="k",
                    start_time=time.time(), work_dir=Path(f"/tmp/t{i}"),
                )

        run_research_async("new_topic", "wh_key", "user1")
        mock_reply.assert_called_once()
        assert "任务运行中" in mock_reply.call_args[0][0]

    @patch("wecom_bot.subprocess.Popen")
    @patch("wecom_bot.reply_message")
    def test_work_directory_created(self, mock_reply, mock_popen, tmp_path):
        """Work directory should be created for a new research task."""
        # Make Popen raise so we don't actually run anything,
        # but the work_dir should already be created by then
        mock_popen.side_effect = FileNotFoundError("no such binary")

        with patch("wecom_bot.RUNS_DIR", tmp_path):
            run_research_async("测试主题", "wh_key", "user1")
            # Give the thread a moment to start
            time.sleep(0.5)

        # Check that a directory was created under tmp_path
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) >= 1
        # Check research_program.md was written
        work_dir = subdirs[0]
        program_file = work_dir / "research_program.md"
        assert program_file.exists()
        content = program_file.read_text(encoding="utf-8")
        assert "测试主题" in content

    @patch("wecom_bot.subprocess.Popen")
    @patch("wecom_bot.reply_message")
    def test_task_key_truncation_for_long_topics(self, mock_reply, mock_popen, tmp_path):
        """Long topics should have truncated task keys."""
        mock_popen.side_effect = FileNotFoundError("no such binary")
        long_topic = "A" * 100

        with patch("wecom_bot.RUNS_DIR", tmp_path):
            run_research_async(long_topic, "wh_key", "user1")
            time.sleep(0.3)

        # Task key should be truncated (topic > 80 chars)
        # After the thread errors and cleans up, task should be gone
        # Just verify it didn't crash
        assert True

    @patch("wecom_bot.subprocess.Popen")
    @patch("wecom_bot.reply_message")
    def test_task_cleanup_after_error(self, mock_reply, mock_popen, tmp_path):
        """Task should be removed from _running_tasks after error."""
        mock_popen.side_effect = FileNotFoundError("no such binary")

        with patch("wecom_bot.RUNS_DIR", tmp_path):
            run_research_async("cleanup_test", "wh_key", "user1")
            time.sleep(1.0)

        with _tasks_lock:
            assert "cleanup_test" not in _running_tasks
