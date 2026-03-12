"""
app.py — Streamlit frontend for autoresearch.

Run: uv run streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
RESEARCH_PROGRAM_PATH = PROJECT_DIR / "research_program.md"
FINDINGS_PATH = PROJECT_DIR / "findings.md"
PROGRESS_PATH = PROJECT_DIR / "progress.tsv"
RUNS_DIR = PROJECT_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(page_title="AutoResearch", page_icon="🔬", layout="wide")
st.title("🔬 AutoResearch")
st.caption("Autonomous research agent — iterative gap analysis, search, and synthesis")

# ── Sidebar config ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    model = st.selectbox(
        "LLM Model",
        ["deepseek-chat", "deepseek-reasoner", "claude-sonnet-4-20250514", "gpt-4o", "gemini-2.0-flash"],
        index=0,
    )
    max_iterations = st.slider("Max Iterations", 1, 20, 5)
    target_coverage = st.slider("Target Coverage", 0.1, 1.0, 0.8, 0.05)
    time_budget = st.slider("Time Budget (min)", 5, 120, 60)

    st.divider()
    st.header("🔑 API Keys")
    st.caption("Keys are stored in session only, never saved to disk.")

    api_keys = {}
    for key_name, label in [
        ("DEEPSEEK_API_KEY", "DeepSeek"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENAI_API_KEY", "OpenAI"),
        ("METASO_API_KEY", "Metaso (秘塔)"),
        ("PERPLEXITY_API_KEY", "Perplexity"),
        ("GOOGLE_API_KEY", "Google/Gemini"),
    ]:
        val = st.text_input(label, value=os.environ.get(key_name, ""), type="password", key=key_name)
        if val:
            api_keys[key_name] = val

    st.divider()
    st.header("🤖 WeCom Bot")
    wecom_webhook = st.text_input(
        "Webhook Key",
        type="password",
        help="The key part of your WeCom bot webhook URL",
    )
    wecom_notify = st.checkbox("Notify on completion", value=bool(wecom_webhook))


# ── WeCom helper ─────────────────────────────────────────────────────────


def send_wecom_message(webhook_key: str, content: str) -> bool:
    """Send a markdown message to a WeCom group bot."""
    if not webhook_key:
        return False
    url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
    # WeCom markdown has a 4096 char limit
    truncated = content[:4000] + "\n\n..." if len(content) > 4000 else content
    try:
        resp = httpx.post(url, json={
            "msgtype": "markdown",
            "markdown": {"content": truncated},
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ── Run history ──────────────────────────────────────────────────────────


def save_run(topic: str, findings: str, progress: str, program: str) -> str:
    """Save a completed run to the runs directory. Returns run ID."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    (run_dir / "findings.md").write_text(findings, encoding="utf-8")
    (run_dir / "progress.tsv").write_text(progress, encoding="utf-8")
    (run_dir / "research_program.md").write_text(program, encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({
        "topic": topic,
        "timestamp": run_id,
        "model": model,
        "max_iterations": max_iterations,
        "target_coverage": target_coverage,
    }, ensure_ascii=False), encoding="utf-8")
    return run_id


def list_runs() -> list[dict]:
    """List all saved runs, newest first."""
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["run_id"] = d.name
            meta["has_findings"] = (d / "findings.md").exists()
            runs.append(meta)
    return runs


# ── Research runner ──────────────────────────────────────────────────────


def run_research(program_text: str, status_container) -> tuple[bool, str]:
    """Run research.py as a subprocess, streaming output to Streamlit."""
    # Write research program
    RESEARCH_PROGRAM_PATH.write_text(program_text, encoding="utf-8")

    # Clear previous outputs
    if FINDINGS_PATH.exists():
        FINDINGS_PATH.unlink()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    # Build environment
    env = os.environ.copy()
    env["AUTORESEARCH_MODEL"] = model
    env["PATH"] = str(Path.home() / ".local" / "bin") + ":" + env.get("PATH", "")
    for k, v in api_keys.items():
        env[k] = v

    cmd = [
        sys.executable, str(PROJECT_DIR / "research.py"),
        "--mode", "auto",
        "--max-iterations", str(max_iterations),
        "--target-coverage", str(target_coverage),
        "--time-budget", str(time_budget),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_DIR),
        env=env,
        bufsize=1,
    )

    output_lines = []
    with status_container.status("🔄 Research running...", expanded=True) as status:
        log_area = st.empty()
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            output_lines.append(line)
            # Show last 30 lines
            display = "\n".join(output_lines[-30:])
            log_area.code(display, language="text")

            # Update status label on key events
            if "Iteration" in line and "Elapsed" in line:
                status.update(label=f"🔄 {line.strip()}")
            elif "Coverage:" in line:
                status.update(label=f"📊 {line.strip()}")
            elif "Research complete" in line:
                status.update(label="✅ Research complete!", state="complete")

        process.wait()

    success = process.returncode == 0
    full_output = "\n".join(output_lines)
    return success, full_output


# ── Main UI ──────────────────────────────────────────────────────────────

tab_new, tab_history = st.tabs(["📝 New Research", "📚 History"])

# ── Tab: New Research ────────────────────────────────────────────────────

with tab_new:
    # Load existing program as default
    existing_program = ""
    if RESEARCH_PROGRAM_PATH.exists():
        existing_program = RESEARCH_PROGRAM_PATH.read_text(encoding="utf-8")

    topic = st.text_input("Research Topic", placeholder="e.g. 潜水蛙鞋设计")

    use_template = st.checkbox("Auto-generate research program from topic", value=True)

    if use_template:
        program_text = st.text_area(
            "Research Program (auto-generated, editable)",
            value=existing_program,
            height=300,
            help="Edit the generated program or write your own",
        )
        if topic and st.button("🪄 Generate from topic"):
            template = f"""# Research Program: {topic}

## Topic
{topic}

## Research Questions
1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]

## Scope
- Include academic and industry sources
- Cover recent developments (2020-2026)

## Constraints
- Prioritize technical depth over breadth
- Include quantitative data where available
"""
            st.session_state["generated_program"] = template
            st.rerun()

        if "generated_program" in st.session_state:
            program_text = st.session_state.pop("generated_program")
    else:
        program_text = st.text_area(
            "Research Program (markdown)",
            value=existing_program,
            height=300,
        )

    # Validation
    has_keys = bool(api_keys)
    has_program = bool(program_text.strip())

    if not has_keys:
        st.warning("Please provide at least one LLM API key in the sidebar.")
    if not has_program:
        st.info("Enter a research topic and program above.")

    col1, col2 = st.columns([1, 4])
    with col1:
        run_clicked = st.button(
            "🚀 Start Research",
            disabled=not (has_keys and has_program),
            type="primary",
            use_container_width=True,
        )

    if run_clicked:
        status_container = st.container()
        success, output = run_research(program_text, status_container)

        if success and FINDINGS_PATH.exists():
            findings = FINDINGS_PATH.read_text(encoding="utf-8")
            progress = PROGRESS_PATH.read_text(encoding="utf-8") if PROGRESS_PATH.exists() else ""

            # Save to history
            run_id = save_run(topic or "Untitled", findings, progress, program_text)
            st.success(f"Research complete! Run saved as `{run_id}`")

            # Show findings
            st.divider()
            st.subheader("📄 Findings")
            st.markdown(findings)

            # Download button
            st.download_button(
                "⬇️ Download findings.md",
                data=findings,
                file_name=f"findings_{run_id}.md",
                mime="text/markdown",
            )

            # WeCom notification
            if wecom_notify and wecom_webhook:
                summary = f"**🔬 AutoResearch Complete**\n\n" \
                          f"**Topic**: {topic}\n" \
                          f"**Model**: {model}\n" \
                          f"**Run ID**: {run_id}\n\n" \
                          f"---\n\n{findings}"
                if send_wecom_message(wecom_webhook, summary):
                    st.toast("✅ WeCom notification sent!")
                else:
                    st.toast("❌ WeCom notification failed", icon="⚠️")
        else:
            st.error("Research failed. Check the log above for errors.")

# ── Tab: History ─────────────────────────────────────────────────────────

with tab_history:
    runs = list_runs()
    if not runs:
        st.info("No research runs yet. Start a new research from the first tab.")
    else:
        for run in runs:
            run_id = run["run_id"]
            run_dir = RUNS_DIR / run_id
            ts = datetime.strptime(run_id, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")

            with st.expander(f"**{run.get('topic', 'Untitled')}** — {ts} ({run.get('model', 'unknown')})"):
                findings_path = run_dir / "findings.md"
                progress_path = run_dir / "progress.tsv"

                if findings_path.exists():
                    findings = findings_path.read_text(encoding="utf-8")

                    # Show progress table
                    if progress_path.exists():
                        progress_text = progress_path.read_text(encoding="utf-8")
                        lines = [l.split("\t") for l in progress_text.strip().split("\n")]
                        if len(lines) > 1:
                            st.caption("Progress")
                            import pandas as pd
                            df = pd.DataFrame(lines[1:], columns=lines[0])
                            st.dataframe(df, use_container_width=True, hide_index=True)

                    # Show findings
                    st.markdown(findings)

                    # Download + WeCom resend
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "⬇️ Download .md",
                            data=findings,
                            file_name=f"findings_{run_id}.md",
                            mime="text/markdown",
                            key=f"dl_{run_id}",
                        )
                    with c2:
                        if wecom_webhook and st.button("📤 Send to WeCom", key=f"wc_{run_id}"):
                            summary = f"**🔬 AutoResearch**\n\n**Topic**: {run.get('topic')}\n\n---\n\n{findings}"
                            if send_wecom_message(wecom_webhook, summary):
                                st.toast("✅ Sent!")
                            else:
                                st.toast("❌ Failed", icon="⚠️")
                else:
                    st.warning("Findings file not found.")
