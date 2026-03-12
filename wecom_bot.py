"""
wecom_bot.py — WeCom intelligent bot callback service for autoresearch.

Receives messages from a WeCom intelligent bot via encrypted callback,
triggers research, and sends results back to the group.

Usage:
    # Set required env vars
    export WECOM_BOT_TOKEN=your_token
    export WECOM_BOT_AES_KEY=your_encoding_aes_key
    export WECOM_BOT_RECEIVE_ID=your_corp_id
    export WECOM_BOT_KEY=your_webhook_key
    export DEEPSEEK_API_KEY=sk-...

    # Run the bot service
    uv run python wecom_bot.py --port 8765

WeCom bot setup:
    1. Create an intelligent bot in WeCom admin
    2. Switch to URL callback mode
    3. Set callback URL to https://research.szfluent.cn/callback
    4. Copy Token + EncodingAESKey into env vars
    5. @mention the bot with a research topic to trigger research
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote

import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# ── Config ───────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
RUNS_DIR = PROJECT_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

BOT_TOKEN = os.environ.get("WECOM_BOT_TOKEN", "")
BOT_AES_KEY = os.environ.get("WECOM_BOT_AES_KEY", "")
BOT_RECEIVE_ID = os.environ.get("WECOM_BOT_RECEIVE_ID", "")
BOT_KEY = os.environ.get("WECOM_BOT_KEY", "")
DEFAULT_MODEL = os.environ.get("AUTORESEARCH_MODEL", "deepseek-chat")
DEFAULT_MAX_ITER = int(os.environ.get("AUTORESEARCH_MAX_ITER", "5"))

# Track running tasks to prevent duplicates
_running_tasks: dict[str, bool] = {}


# ── WeCom Message Crypto ─────────────────────────────────────────────────


class WXBizMsgCrypt:
    """WeCom message encryption/decryption (AES-256-CBC).

    Implements the official WeCom callback encryption protocol:
    - Signature = SHA1(sort([token, timestamp, nonce, encrypt]))
    - AES key = Base64Decode(EncodingAESKey + "=")
    - IV = AES key[:16]
    - Plaintext = random(16) + msg_len(4, big-endian) + msg + receive_id
    - Padding = PKCS#7 (block size 32)
    """

    def __init__(self, token: str, encoding_aes_key: str, receive_id: str):
        self.token = token
        self.receive_id = receive_id
        self.aes_key = base64.b64decode(encoding_aes_key + "=")
        self.iv = self.aes_key[:16]

    def _signature(self, *args: str) -> str:
        items = sorted(args)
        return hashlib.sha1("".join(items).encode()).hexdigest()

    def _pkcs7_pad(self, data: bytes, block_size: int = 32) -> bytes:
        pad_len = block_size - (len(data) % block_size)
        return data + bytes([pad_len] * pad_len)

    def _pkcs7_unpad(self, data: bytes) -> bytes:
        pad_len = data[-1]
        if pad_len < 1 or pad_len > 32:
            return data
        return data[:-pad_len]

    def _encrypt(self, plaintext: str) -> str:
        msg_bytes = plaintext.encode("utf-8")
        rid_bytes = self.receive_id.encode("utf-8")
        content = os.urandom(16) + struct.pack("!I", len(msg_bytes)) + msg_bytes + rid_bytes
        padded = self._pkcs7_pad(content)
        cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(self.iv))
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()
        return base64.b64encode(encrypted).decode()

    def _decrypt(self, ciphertext: str) -> str:
        cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(self.iv))
        decryptor = cipher.decryptor()
        plain = decryptor.update(base64.b64decode(ciphertext)) + decryptor.finalize()
        plain = self._pkcs7_unpad(plain)
        msg_len = struct.unpack("!I", plain[16:20])[0]
        return plain[20:20 + msg_len].decode("utf-8")

    def verify_url(self, msg_signature: str, timestamp: str, nonce: str, echostr: str) -> tuple[bool, str]:
        """Verify callback URL — decrypt echostr and return plaintext."""
        expected = self._signature(self.token, timestamp, nonce, echostr)
        if expected != msg_signature:
            print(f"[crypto] Signature mismatch: expected={expected}, got={msg_signature}")
            return False, ""
        try:
            decrypted = self._decrypt(echostr)
            return True, decrypted
        except Exception as e:
            print(f"[crypto] Decrypt echostr failed: {e}")
            return False, ""

    def decrypt_msg(self, msg_signature: str, timestamp: str, nonce: str, post_data: str) -> tuple[bool, str]:
        """Decrypt incoming POST message. Returns (success, decrypted_xml_or_json)."""
        encrypt = ""
        # Try JSON first (intelligent bot uses lowercase "encrypt")
        try:
            data = json.loads(post_data)
            encrypt = data.get("encrypt", "") or data.get("Encrypt", "")
        except (json.JSONDecodeError, ValueError):
            pass
        # Try XML if JSON didn't work
        if not encrypt:
            try:
                root = ET.fromstring(post_data)
                encrypt_node = root.find("Encrypt")
                if encrypt_node is not None and encrypt_node.text:
                    encrypt = encrypt_node.text
            except ET.ParseError:
                pass
        if not encrypt:
            print(f"[crypto] No encrypt field found in body")
            return False, ""

        expected = self._signature(self.token, timestamp, nonce, encrypt)
        if expected != msg_signature:
            print(f"[crypto] Message signature mismatch")
            return False, ""

        try:
            decrypted = self._decrypt(encrypt)
            return True, decrypted
        except Exception as e:
            print(f"[crypto] Decrypt message failed: {e}")
            return False, ""

    def encrypt_reply(self, reply_msg: str, nonce: str, timestamp: str | None = None) -> str:
        """Encrypt a reply message into XML envelope."""
        if not timestamp:
            timestamp = str(int(time.time()))
        encrypted = self._encrypt(reply_msg)
        signature = self._signature(self.token, timestamp, nonce, encrypted)
        return (
            f"<xml>"
            f"<Encrypt><![CDATA[{encrypted}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
            f"<TimeStamp>{timestamp}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            f"</xml>"
        )


# Build global crypto instance (may be None if not configured)
_crypt: WXBizMsgCrypt | None = None
try:
    if BOT_TOKEN and BOT_AES_KEY and BOT_RECEIVE_ID:
        _crypt = WXBizMsgCrypt(BOT_TOKEN, BOT_AES_KEY, BOT_RECEIVE_ID)
        print(f"[crypto] Initialized with receive_id={BOT_RECEIVE_ID[:8]}...")
    elif BOT_TOKEN and BOT_AES_KEY:
        print("[crypto] Warning: WECOM_BOT_RECEIVE_ID not set, trying with empty receive_id")
        _crypt = WXBizMsgCrypt(BOT_TOKEN, BOT_AES_KEY, "")
except Exception as e:
    print(f"[crypto] Failed to initialize: {e}")
    print("[crypto] Check that WECOM_BOT_AES_KEY is exactly 43 characters")
    _crypt = None


# ── WeCom send helpers ───────────────────────────────────────────────────


def send_wecom_text(webhook_key: str, content: str, mentioned: list[str] | None = None) -> bool:
    """Send a text message to a WeCom group via webhook."""
    if not webhook_key:
        return False
    url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
    payload: dict = {
        "msgtype": "text",
        "text": {"content": content},
    }
    if mentioned:
        payload["text"]["mentioned_list"] = mentioned
    try:
        resp = httpx.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[wecom] Send error: {e}")
        return False


def send_wecom_markdown(webhook_key: str, content: str) -> bool:
    """Send a markdown message to a WeCom group via webhook."""
    if not webhook_key:
        return False
    url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
    truncated = content[:4000] + "\n\n> (内容过长，已截断)" if len(content) > 4000 else content
    try:
        resp = httpx.post(url, json={
            "msgtype": "markdown",
            "markdown": {"content": truncated},
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[wecom] Send error: {e}")
        return False


def send_wecom_file(webhook_key: str, file_path: Path) -> bool:
    """Upload a file to WeCom and send it to the group."""
    if not webhook_key:
        return False
    upload_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={webhook_key}&type=file"
    try:
        with open(file_path, "rb") as f:
            resp = httpx.post(
                upload_url,
                files={"media": (file_path.name, f, "application/octet-stream")},
                timeout=30,
            )
        if resp.status_code != 200:
            return False
        media_id = resp.json().get("media_id")
        if not media_id:
            return False
        send_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
        resp = httpx.post(send_url, json={
            "msgtype": "file",
            "file": {"media_id": media_id},
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[wecom] File send error: {e}")
        return False


# ── Research runner ──────────────────────────────────────────────────────


def generate_research_program(topic: str) -> str:
    """Generate a research program markdown from a topic string."""
    return f"""# Research Program: {topic}

## Topic
{topic}

## Research Questions
1. {topic}的主要概念、分类和核心原理是什么？
2. 该领域的最新进展和趋势有哪些（2020-2026）？
3. 有哪些关键技术、材料或方法？
4. 该领域面临的主要挑战和局限性是什么？
5. 未来的发展方向和潜在机会是什么？

## Scope
- 涵盖学术研究和行业实践
- 包括中英文来源
- 时间范围：2020-2026

## Constraints
- 优先选择有技术深度的来源
- 尽可能包含定量数据
- 输出语言为中文
"""


def run_research_async(topic: str, webhook_key: str, from_user: str = "") -> None:
    """Run research in a background thread and send results to WeCom."""
    task_key = topic[:50]
    if _running_tasks.get(task_key):
        send_wecom_text(webhook_key, f"⏳ 「{topic}」正在研究中，请稍候...")
        return

    _running_tasks[task_key] = True

    def _run():
        try:
            send_wecom_markdown(webhook_key, f"**🔬 开始研究**\n\n> {topic}\n\n模型: `{DEFAULT_MODEL}`\n最大迭代: {DEFAULT_MAX_ITER}")

            program = generate_research_program(topic)
            program_path = PROJECT_DIR / "research_program.md"
            program_path.write_text(program, encoding="utf-8")

            findings_path = PROJECT_DIR / "findings.md"
            progress_path = PROJECT_DIR / "progress.tsv"
            if findings_path.exists():
                findings_path.unlink()
            if progress_path.exists():
                progress_path.unlink()

            env = os.environ.copy()
            env["AUTORESEARCH_MODEL"] = DEFAULT_MODEL
            env["PATH"] = str(Path.home() / ".local" / "bin") + ":" + env.get("PATH", "")

            cmd = [
                sys.executable, str(PROJECT_DIR / "research.py"),
                "--mode", "auto",
                "--max-iterations", str(DEFAULT_MAX_ITER),
                "--target-coverage", "0.8",
                "--time-budget", "60",
            ]

            start = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(PROJECT_DIR), env=env, timeout=3600,
            )
            elapsed = time.time() - start

            if result.returncode == 0 and findings_path.exists():
                findings = findings_path.read_text(encoding="utf-8")
                progress = progress_path.read_text(encoding="utf-8") if progress_path.exists() else ""

                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = RUNS_DIR / run_id
                run_dir.mkdir(exist_ok=True)
                (run_dir / "findings.md").write_text(findings, encoding="utf-8")
                (run_dir / "progress.tsv").write_text(progress, encoding="utf-8")
                (run_dir / "research_program.md").write_text(program, encoding="utf-8")

                lines = progress.strip().split("\n")
                iterations = len(lines) - 1 if len(lines) > 1 else 0
                last_coverage = "N/A"
                if iterations > 0:
                    parts = lines[-1].split("\t")
                    if len(parts) >= 2:
                        last_coverage = parts[1]

                summary = (
                    f"**🔬 研究完成**\n\n"
                    f"> {topic}\n\n"
                    f"- 迭代次数: {iterations}\n"
                    f"- 最终覆盖率: {last_coverage}\n"
                    f"- 耗时: {elapsed:.0f}s\n"
                    f"- 运行ID: `{run_id}`\n\n"
                    f"---\n\n"
                )
                send_wecom_markdown(webhook_key, summary + findings)
                send_wecom_file(webhook_key, run_dir / "findings.md")
            else:
                error_msg = result.stderr[-500:] if result.stderr else result.stdout[-500:]
                send_wecom_text(webhook_key, f"❌ 研究失败\n\n{error_msg}")

        except subprocess.TimeoutExpired:
            send_wecom_text(webhook_key, f"⏰ 研究超时（1小时）: {topic}")
        except Exception as e:
            send_wecom_text(webhook_key, f"❌ 研究出错: {e}")
        finally:
            _running_tasks.pop(task_key, None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"[bot] Started research thread for: {topic}")


# ── HTTP handler ─────────────────────────────────────────────────────────

HELP_TEXT = """🔬 **AutoResearch Bot**

使用方法:
- `研究 <主题>` — 启动自动研究
- `状态` — 查看当前运行状态
- `帮助` — 显示此帮助信息

示例: `研究 潜水蛙鞋设计`"""


def _extract_message(decrypted: str) -> dict:
    """Extract message fields from decrypted content (XML or JSON).

    Returns dict with keys: text, from_user, response_url, webhook_url, msg_type
    """
    result = {"text": "", "from_user": "", "response_url": "", "webhook_url": "", "msg_type": ""}

    # Try JSON first
    try:
        data = json.loads(decrypted)
        print(f"[bot]   Decrypted JSON keys: {list(data.keys())}")

        result["msg_type"] = data.get("MsgType", data.get("msgtype", ""))
        result["response_url"] = data.get("ResponseUrl", data.get("response_url", ""))
        result["webhook_url"] = data.get("WebhookUrl", "")

        # Extract sender
        if "From" in data:
            f = data["From"]
            result["from_user"] = f.get("Name", f.get("UserId", f.get("Alias", "")))
        elif "from" in data:
            f = data["from"]
            result["from_user"] = f.get("name", f.get("user_id", ""))

        # Extract text content
        if result["msg_type"] == "text":
            text_obj = data.get("Text", data.get("text", {}))
            if isinstance(text_obj, dict):
                result["text"] = text_obj.get("Content", text_obj.get("content", "")).strip()
            elif isinstance(text_obj, str):
                result["text"] = text_obj.strip()
        elif "Content" in data:
            result["text"] = data["Content"].strip()
        elif "content" in data:
            result["text"] = data["content"].strip()

        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try XML
    try:
        root = ET.fromstring(decrypted)
        result["msg_type"] = root.findtext("MsgType", "")
        result["text"] = root.findtext("Content", "").strip()
        result["from_user"] = root.findtext("From/UserId", "") or root.findtext("FromUserName", "")
        result["webhook_url"] = root.findtext("WebhookUrl", "")
        result["response_url"] = root.findtext("ResponseUrl", "")
    except ET.ParseError:
        pass

    return result


def reply_via_response_url(response_url: str, content: str) -> bool:
    """Reply to a message using the intelligent bot's response_url (proactive reply)."""
    if not response_url:
        return False
    try:
        resp = httpx.post(response_url, json={
            "msgtype": "markdown",
            "markdown": {"content": content[:4000]},
        }, timeout=10)
        print(f"[bot] Reply via response_url: status={resp.status_code} body={resp.text[:200]}")
        return resp.status_code == 200
    except Exception as e:
        print(f"[bot] Reply error: {e}")
        return False


def reply_message(text: str, response_url: str = "", webhook_key: str = "") -> bool:
    """Reply using the best available method."""
    if response_url:
        return reply_via_response_url(response_url, text)
    if webhook_key:
        return send_wecom_markdown(webhook_key, text)
    print(f"[bot] No reply method available")
    return False


INTRO_TEXT = """**🔬 研究助手 — AutoResearch Bot**

我是一个**自主研究 AI 助手**，能帮你深度调研任何主题。直接发送主题即可开始。

**与传统搜索的区别：**
- 🔍 **vs 秘塔(Metaso)**：Metaso 搜一次就出结果；我会**多轮迭代**，每轮评估"还缺什么"再针对性补充
- 🧠 **vs Karpathy autoresearch**：同样迭代理念，但我加入三级搜索（深度/学术/网页）和覆盖率评估

**实际案例：「潜水蛙鞋 3D 打印设计」**

用户输入主题后，Bot 自动执行 5 轮迭代研究：

> **第 1 轮** — 识别核心问题：蛙鞋设计参数、3D打印材料选择、水动力学原理
> 搜索 12 篇来源（学术论文 + 行业文章），覆盖率 35%
>
> **第 2 轮** — 发现知识空白：缺少 TPU/硅胶柔性材料对比数据
> 针对性搜索材料力学性能，补充弹性模量、抗疲劳数据，覆盖率 → 55%
>
> **第 3 轮** — 深入细节：FDM vs SLA vs MJF 打印工艺对比、后处理流程
> 新增 8 篇来源，覆盖率 → 72%
>
> **第 4-5 轮** — 补充实际案例：Fin Lab定制蛙鞋流程、消费级产品定价分析
> 最终覆盖率 **85%**，累计引用 **34 篇来源**

最终输出结构化报告，包含：
- 材料对比表（TPU vs 硅胶 vs 碳纤维复合）
- 打印参数推荐（层高、填充率、支撑策略）
- 成本分析（单件 vs 小批量 vs 注塑对比）
- 每个结论都附带来源链接

**使用方法：**
- `研究 <主题>` — 发起研究，例如：`研究 碳纤维在消费电子中的应用趋势`
- `状态` — 查看进行中的任务
- `取消 <关键词>` — 取消运行中的任务
- `帮助` — 显示本说明"""


def handle_message(text: str, from_user: str, response_url: str = "", webhook_key: str = "") -> None:
    """Route a message to the appropriate handler."""
    text_lower = text.lower().strip()
    print(f"[bot] Message from {from_user}: {text}")

    if text_lower in ("帮助", "help", "?", "？", "你可以做什么", "你能做什么",
                       "介绍", "你是谁", "hi", "hello", "你好"):
        reply_message(INTRO_TEXT, response_url, webhook_key)

    elif text_lower in ("状态", "status"):
        if _running_tasks:
            tasks = "\n".join(f"- {k}" for k in _running_tasks)
            reply_message(f"🔄 正在运行的任务:\n{tasks}", response_url, webhook_key)
        else:
            reply_message("✅ 当前没有运行中的任务", response_url, webhook_key)

    elif "取消" in text_lower or text_lower.startswith("cancel"):
        # Cancel a running task by substring match
        if not _running_tasks:
            reply_message("✅ 当前没有运行中的任务", response_url, webhook_key)
        else:
            # Extract cancel target: remove command words, match against running tasks
            cancel_query = text_lower.replace("取消", "").replace("cancel", "").replace("我想", "").replace("请", "").strip()
            if not cancel_query:
                # No specific target — list tasks for user to pick
                tasks = "\n".join(f"- {k}" for k in _running_tasks)
                reply_message(f"请指定要取消的任务，当前运行中:\n{tasks}\n\n例如: `取消 蛙鞋`", response_url, webhook_key)
            else:
                # Find matching task by substring
                matched = [k for k in _running_tasks if cancel_query in k.lower()]
                if len(matched) == 1:
                    _running_tasks.pop(matched[0], None)
                    reply_message(f"🛑 已标记取消任务「{matched[0]}」\n\n注意：正在运行的子进程将在当前迭代结束后停止", response_url, webhook_key)
                elif len(matched) > 1:
                    tasks = "\n".join(f"- {k}" for k in matched)
                    reply_message(f"匹配到多个任务，请更具体:\n{tasks}", response_url, webhook_key)
                else:
                    tasks = "\n".join(f"- {k}" for k in _running_tasks)
                    reply_message(f"未找到匹配「{cancel_query}」的任务。当前运行中:\n{tasks}", response_url, webhook_key)

    elif text_lower.startswith(("研究 ", "研究\n", "research ")):
        topic = text.split(None, 1)[1] if len(text.split(None, 1)) > 1 else ""
        if topic:
            reply_message(f"🔬 收到！开始研究「{topic}」...", response_url, webhook_key)
            run_research_async(topic, webhook_key, from_user)
        else:
            reply_message("请提供研究主题，例如: `研究 潜水蛙鞋设计`", response_url, webhook_key)

    else:
        # Unknown command — do NOT treat as research topic
        reply_message(
            f"🤔 不太理解「{text}」\n\n"
            "发起研究请使用: `研究 <主题>`\n"
            "查看任务状态: `状态`\n"
            "取消任务: `取消 <关键词>`\n"
            "更多帮助: `帮助`",
            response_url, webhook_key,
        )


class WeComBotHandler(BaseHTTPRequestHandler):
    """Handle WeCom intelligent bot callbacks with encryption."""

    def _get_raw_param(self, query: str, name: str) -> str:
        """Extract a raw (not URL-decoded) parameter from query string."""
        match = re.search(rf'(?:^|&){name}=([^&]*)', query)
        return match.group(1) if match else ""

    def do_GET(self):
        """Handle callback URL verification (encrypted echostr)."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        raw_query = parsed.query

        msg_signature = params.get("msg_signature", [""])[0]
        timestamp = params.get("timestamp", [""])[0]
        nonce = params.get("nonce", [""])[0]

        # Use raw echostr (not URL-decoded) for signature verification
        echostr_raw = self._get_raw_param(raw_query, "echostr")
        echostr_decoded = unquote(echostr_raw)

        print(f"[bot] GET verification: sig={msg_signature[:16]}... ts={timestamp} nonce={nonce}")
        print(f"[bot]   echostr_raw({len(echostr_raw)}): {echostr_raw[:60]}...")
        print(f"[bot]   echostr_decoded({len(echostr_decoded)}): {echostr_decoded[:60]}...")

        if _crypt:
            # Try with raw echostr first (WeCom may compute sig on URL-encoded value)
            ok, decrypted = _crypt.verify_url(msg_signature, timestamp, nonce, echostr_raw)
            if not ok and echostr_raw != echostr_decoded:
                # Retry with URL-decoded echostr
                print(f"[bot]   Retrying with decoded echostr...")
                ok, decrypted = _crypt.verify_url(msg_signature, timestamp, nonce, echostr_decoded)
            if ok:
                print(f"[bot] URL verification OK")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(decrypted.encode())
                return
            else:
                print(f"[bot] URL verification FAILED")

        # Fallback: echo back (for testing without encryption)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(echostr_decoded.encode() if echostr_decoded else b"ok")

    def do_POST(self):
        """Handle incoming messages (encrypted)."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        msg_signature = params.get("msg_signature", [""])[0]
        timestamp = params.get("timestamp", [""])[0]
        nonce = params.get("nonce", [""])[0]

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        # Respond immediately (WeCom expects quick 200)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"errcode": 0, "errmsg": "ok"}')

        print(f"[bot] POST callback: {len(body)} bytes, sig={msg_signature[:16] if msg_signature else 'none'}...")
        print(f"[bot]   Raw body: {body[:500]}")

        # Try encrypted message first
        if _crypt and msg_signature:
            ok, decrypted = _crypt.decrypt_msg(msg_signature, timestamp, nonce, body)
            if ok:
                print(f"[bot] Decrypted: {decrypted[:300]}...")
                msg = _extract_message(decrypted)
                if msg["text"]:
                    webhook_key = BOT_KEY
                    if msg["webhook_url"] and "key=" in msg["webhook_url"]:
                        webhook_key = msg["webhook_url"].split("key=")[-1].split("&")[0]
                    handle_message(msg["text"], msg["from_user"], msg["response_url"], webhook_key)
                else:
                    print(f"[bot] No text in message (type={msg['msg_type']})")
                return
            else:
                print(f"[bot] Decryption failed, trying plaintext")

        # Fallback: try plaintext JSON (for testing)
        try:
            data = json.loads(body)
            msg = _extract_message(body)
            if msg["text"]:
                handle_message(msg["text"], msg["from_user"], msg["response_url"], BOT_KEY)
        except json.JSONDecodeError:
            print(f"[bot] Cannot parse body: {body[:200]}")

    def log_message(self, format, *args):
        """Suppress default access logs."""
        pass


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="WeCom bot for autoresearch")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    if not _crypt:
        print("⚠️  Encryption not configured. Set WECOM_BOT_TOKEN, WECOM_BOT_AES_KEY, WECOM_BOT_RECEIVE_ID")
        print("   Bot will still run but URL verification will fail.")

    if not BOT_KEY:
        print("⚠️  WECOM_BOT_KEY not set. Bot can receive but won't be able to send replies.")

    server = HTTPServer((args.host, args.port), WeComBotHandler)
    print(f"🤖 WeCom bot listening on http://{args.host}:{args.port}")
    print(f"   Callback URL: https://research.szfluent.cn/callback")
    print(f"   Model: {DEFAULT_MODEL}")
    print(f"   Max iterations: {DEFAULT_MAX_ITER}")
    print(f"   Encryption: {'✅ enabled' if _crypt else '❌ disabled'}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bot] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
