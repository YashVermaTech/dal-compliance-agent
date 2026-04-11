"""
DAL-Aware Compliance Agent — Streamlit Frontend (HuggingFace Spaces)
=====================================================================
5-tab dark-themed dashboard for aerospace DO-178C certification.

Tabs:
  1. Document Ingestion
  2. Compliance Chat (EN + DE)
  3. Traceability Matrix
  4. Gap Analysis
  5. Impact Analyzer

Author: Yash Verma — AI Engineer, TU Darmstadt
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
API = f"{BACKEND_URL}/api/v1"

DAL_INFO = {
    "A": {"label": "DAL-A — Catastrophic", "color": "#f85149", "bg": "#3d1a1a", "objectives": 71, "coverage": "MC/DC"},
    "B": {"label": "DAL-B — Hazardous",    "color": "#f0a500", "bg": "#3d2b00", "objectives": 69, "coverage": "Decision"},
    "C": {"label": "DAL-C — Major",        "color": "#f0e000", "bg": "#3d3900", "objectives": 62, "coverage": "Statement"},
    "D": {"label": "DAL-D — Minor",        "color": "#3fb950", "bg": "#1a4731", "objectives": 26, "coverage": "Statement"},
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DAL-Aware Compliance Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
.stApp { background-color: #0d1117; color: #e6edf3; }
.main .block-container { padding-top: 1.2rem; max-width: 1200px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #161b22 0%, #0f3460 100%);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.4rem;
}
.app-header h1 { color: #58a6ff; margin: 0; font-size: 1.7rem; font-weight: 700; }
.app-header p  { color: #8b949e; margin: 0.25rem 0 0 0; font-size: 0.85rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 6px; padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #e6edf3; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 2px solid #30363d; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 500; padding: 0.5rem 1rem; }
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* ── Buttons ── */
.stButton > button {
    background: #238636; color: #fff; border: none;
    border-radius: 6px; font-weight: 600; padding: 0.4rem 1.2rem;
}
.stButton > button:hover { background: #2ea043; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 6px;
}

/* ── Chat bubbles ── */
.chat-user {
    background: #1f2937; border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0; padding: 0.7rem 1rem; margin: 0.4rem 0;
}
.chat-agent {
    background: #161b22; border-left: 3px solid #3fb950;
    border-radius: 0 6px 6px 0; padding: 0.7rem 1rem; margin: 0.4rem 0;
}
.citation-box {
    background: #0d2137; border: 1px solid #1f6feb;
    border-radius: 5px; padding: 0.45rem 0.7rem;
    margin-top: 0.4rem; font-size: 0.81rem; color: #79c0ff;
}

/* ── Status badges ── */
.badge { padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
.badge-green  { background: #1a4731; color: #3fb950; }
.badge-yellow { background: #3d2b00; color: #f0c000; }
.badge-red    { background: #3d1a1a; color: #f85149; }
.badge-blue   { background: #0d2137; color: #58a6ff; }

/* ── Risk labels ── */
.risk-HIGH   { color: #f85149; font-weight: 700; font-size: 1.1rem; }
.risk-MEDIUM { color: #f0a500; font-weight: 700; font-size: 1.1rem; }
.risk-LOW    { color: #3fb950; font-weight: 700; font-size: 1.1rem; }

/* ── DataFrames ── */
.stDataFrame { border: 1px solid #30363d; border-radius: 6px; }

/* ── Dividers ── */
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:12]
if "dal_level" not in st.session_state:
    st.session_state.dal_level = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "last_traceability" not in st.session_state:
    st.session_state.last_traceability = None
if "last_gap" not in st.session_state:
    st.session_state.last_gap = None
if "last_impact" not in st.session_state:
    st.session_state.last_impact = None

SID = st.session_state.session_id

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API}/{endpoint}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot reach backend at `{BACKEND_URL}`. Is it running?")
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out (>120s).")
    except requests.exceptions.HTTPError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None


def _get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API}/{endpoint}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error: {e}")
    return None


def _set_dal(dal: str) -> bool:
    r = _post("set-dal", json={"session_id": SID, "dal_level": dal})
    if r:
        st.session_state.dal_level = dal
        return True
    return False


def _dal_badge(dal: str | None) -> str:
    if not dal:
        return '<span class="badge badge-blue">DAL: Not Set</span>'
    cls = {"A": "badge-red", "B": "badge-yellow", "C": "badge-yellow", "D": "badge-green"}.get(dal, "badge-blue")
    return f'<span class="badge {cls}">DAL-{dal}</span>'


def _status_badge(status: str) -> str:
    cls = {"COVERED": "badge-green", "PARTIAL": "badge-yellow", "MISSING": "badge-red"}.get(status, "badge-blue")
    return f'<span class="badge {cls}">{status}</span>'

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0">
      <div style="font-size:2.8rem">✈️</div>
      <h2 style="color:#58a6ff;margin:0">DAL Compliance</h2>
      <p style="color:#8b949e;font-size:0.75rem;margin:0.2rem 0 0">v1.0.0</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown(f"**Session:** `{SID[:8]}…`")
    st.markdown(f"**Active DAL:** {_dal_badge(st.session_state.dal_level)}",
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### Set DAL Level")
    dal_choice = st.radio("DAL", ["A", "B", "C", "D"], horizontal=True,
                          label_visibility="collapsed")
    if dal_choice in DAL_INFO:
        info = DAL_INFO[dal_choice]
        st.markdown(
            f'<div style="background:{info["bg"]};border-radius:6px;padding:0.5rem 0.7rem;'
            f'border-left:3px solid {info["color"]};font-size:0.82rem;margin:0.3rem 0">'
            f'<strong style="color:{info["color"]}">DAL-{dal_choice}</strong><br>'
            f'{info["objectives"]} objectives · {info["coverage"]} coverage</div>',
            unsafe_allow_html=True,
        )
    if st.button("Apply DAL", use_container_width=True):
        if _set_dal(dal_choice):
            st.success(f"DAL-{dal_choice} applied ✓")
            st.rerun()

    st.divider()
    st.markdown("### Standards")
    for s in ["DO-178C", "DO-254", "ARP4761", "ARP4754A", "DO-160"]:
        st.markdown(f"• `{s}`")

    st.divider()
    if st.button("🔍 Health Check", use_container_width=True):
        h = _get("health")
        if h:
            for svc, info in h.get("components", {}).items():
                icon = "✅" if info.get("status") == "ok" else "⚠️"
                st.markdown(f"{icon} **{svc}**: {info.get('status')}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="app-header">
  <h1>✈️ DAL-Aware Compliance Agent</h1>
  <p>DO-178C · DO-254 · ARP4761 · ARP4754A · DO-160 &nbsp;|&nbsp;
     Session: <code>{SID[:8]}</code> &nbsp;|&nbsp;
     {_dal_badge(st.session_state.dal_level)}</p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Document Ingestion",
    "💬 Compliance Chat",
    "🔗 Traceability Matrix",
    "🔍 Gap Analysis",
    "⚡ Impact Analyzer",
])

# ============================================================
# TAB 1 — DOCUMENT INGESTION
# ============================================================

with tab1:
    st.subheader("Upload & Index Compliance Documents")
    st.caption("Supported: DO-178C, DO-254, DO-160, ARP4761, ARP4754A — PDF or TXT")

    col_up, col_list = st.columns([3, 2])

    with col_up:
        uploaded = st.file_uploader(
            "Drop PDF/TXT files here",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        dal_tag = st.selectbox(
            "DAL tag override",
            ["ALL (auto-detect)", "A", "B", "C", "D"],
            help="'ALL' lets the system detect DAL per chunk from document content.",
        )
        dal_val = "ALL" if dal_tag.startswith("ALL") else dal_tag

        if st.button("⬆️ Index Documents", disabled=not uploaded, use_container_width=True):
            results = []
            bar = st.progress(0)
            for i, uf in enumerate(uploaded):
                with st.spinner(f"Indexing {uf.name}…"):
                    r = requests.post(
                        f"{API}/ingest",
                        files={"file": (uf.name, uf.read(), "application/octet-stream")},
                        data={"dal_level": dal_val, "session_id": SID},
                        timeout=180,
                    )
                if r.status_code in (200, 202):
                    d = r.json()
                    results.append({
                        "File": d["filename"], "Standard": d.get("standard", "?"),
                        "Chunks": d.get("chunks_indexed", 0), "Status": "✅ Indexed",
                    })
                else:
                    results.append({
                        "File": uf.name, "Standard": "—",
                        "Chunks": 0, "Status": f"❌ {r.status_code}",
                    })
                bar.progress((i + 1) / len(uploaded))

            st.success(f"Processed {len(uploaded)} file(s)")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    with col_list:
        st.markdown("#### Index Status")
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.doc_stats = _get("documents")

        stats = getattr(st.session_state, "doc_stats", None) or _get("documents")
        if stats:
            st.session_state.doc_stats = stats
            st.metric("Total Vectors", stats.get("total_vectors", 0))
            st.metric("Index", stats.get("index", "—"))
            st.markdown(
                f'Index status: {_dal_badge(None) if stats.get("status") != "ok" else "<span class=\"badge badge-green\">ready</span>"}',
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload a document to get started.")

# ============================================================
# TAB 2 — COMPLIANCE CHAT
# ============================================================

with tab2:
    st.subheader("Compliance Q&A")
    st.caption("Ask in English or German — answers include section-level citations.")

    # Controls row
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        chat_dal = st.selectbox(
            "Active DAL",
            ["Not Set", "A — Catastrophic", "B — Hazardous", "C — Major", "D — Minor"],
            index=(0 if not st.session_state.dal_level
                   else ["A","B","C","D"].index(st.session_state.dal_level) + 1),
            key="chat_dal",
        )
        if chat_dal != "Not Set":
            new_dal = chat_dal[0]
            if new_dal != st.session_state.dal_level:
                _set_dal(new_dal)
    with c2:
        std_filter = st.selectbox(
            "Standard filter",
            ["All", "DO-178C", "DO-254", "ARP4761", "ARP4754A", "DO-160"],
            key="chat_std",
        )
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # Chat history
    for turn in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-user">👤 <strong>You:</strong> {turn["q"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-agent">🤖 <strong>Agent:</strong> {turn["a"]}</div>',
            unsafe_allow_html=True,
        )
        if turn.get("citations"):
            cits = "".join(
                f"<div>📌 [{c['index']}] <strong>{c['document']}</strong> — "
                f"Section {c['section']}, p.{c['page']}</div>"
                for c in turn["citations"]
            )
            st.markdown(f'<div class="citation-box">Citations:<br>{cits}</div>',
                        unsafe_allow_html=True)
        meta = f"⚡ {turn.get('ms',0)}ms"
        if turn.get("conf"):
            meta += f" · Confidence {turn['conf']:.0%}"
        if turn.get("dal"):
            meta += f" · DAL-{turn['dal']}"
        st.markdown(f"<small style='color:#6e7681'>{meta}</small>", unsafe_allow_html=True)
        st.divider()

    # Example questions
    examples = [
        "What MC/DC coverage is required for DAL-A software?",
        "Welche Verifikationsziele gelten für DAL-B nach DO-178C?",
        "What independence is required for SQA in DAL-A?",
        "What documents must be produced for DO-178C DAL-B certification?",
        "Erkläre den Unterschied zwischen HLR und LLR in DO-178C",
        "What is required for structural coverage at DAL-C?",
    ]
    picked = st.selectbox("Example questions", ["— type below or pick —"] + examples,
                          key="ex_pick")
    question = st.text_area(
        "Your question:",
        value="" if picked.startswith("—") else picked,
        height=80,
        placeholder="Ask any DO-178C / DO-254 / ARP4761 compliance question…",
        key="chat_q",
    )

    if st.button("Ask Agent 🚀", use_container_width=True, type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Agent thinking…"):
                payload: dict[str, Any] = {
                    "session_id": SID,
                    "question": question.strip(),
                }
                if st.session_state.dal_level:
                    payload["dal_level"] = st.session_state.dal_level
                if std_filter != "All":
                    payload["standard"] = std_filter

                result = _post("query", json=payload)

            if result:
                st.session_state.chat_history.append({
                    "q": question.strip(),
                    "a": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "ms": result.get("latency_ms", 0),
                    "conf": result.get("confidence_score", 0),
                    "dal": result.get("dal_level"),
                })
                st.rerun()

# ============================================================
# TAB 3 — TRACEABILITY MATRIX
# ============================================================

with tab3:
    st.subheader("DO-178C Traceability Matrix")
    st.caption("Maps each requirement → verification method → test case → coverage level")

    req_file = st.file_uploader("Upload requirements document (TXT/PDF)",
                                type=["txt","pdf"], key="req_up")
    req_text = st.text_area(
        "Or paste requirements here:",
        height=130,
        placeholder=(
            "REQ-001: The system shall compute altitude within ±1 m accuracy\n"
            "REQ-002: The system shall detect sensor faults within 100 ms\n"
            "REQ-003: The software shall not exceed 80% CPU utilization"
        ),
        key="req_txt",
    )

    if st.button("Generate Traceability Matrix", use_container_width=True, type="primary"):
        content = ""
        name = "requirements.txt"
        if req_file:
            raw = req_file.read()
            content = raw.decode("utf-8", errors="ignore") if req_file.name.endswith(".txt") else ""
            name = req_file.name
            if not content:
                import io, PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                content = "\n".join(p.extract_text() or "" for p in reader.pages)
        elif req_text.strip():
            content = req_text.strip()

        if not content:
            st.warning("Provide a requirements file or paste text.")
        else:
            with st.spinner("Generating matrix…"):
                result = _post("traceability", json={
                    "session_id": SID,
                    "requirements_text": content,
                    "document_name": name,
                })
            if result:
                st.session_state.last_traceability = result

    if st.session_state.last_traceability:
        data = st.session_state.last_traceability
        s = data.get("summary", {})

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",    s.get("total_requirements", 0))
        c2.metric("Covered ✅", s.get("covered", 0))
        c3.metric("Partial ⚠️", s.get("partial", 0))
        c4.metric("Missing ❌", s.get("missing", 0))
        c5.metric("Coverage",  f"{s.get('coverage_percentage', 0)}%")

        # Pie chart
        pie = px.pie(
            pd.DataFrame({
                "Status": ["Covered", "Partial", "Missing"],
                "Count":  [s.get("covered",0), s.get("partial",0), s.get("missing",0)],
            }),
            names="Status", values="Count",
            color="Status",
            color_discrete_map={"Covered":"#3fb950","Partial":"#f0c000","Missing":"#f85149"},
            title="Coverage Distribution",
        )
        pie.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3",
                          title_font_color="#58a6ff")
        st.plotly_chart(pie, use_container_width=True)

        # Table
        matrix = data.get("traceability_matrix", [])
        if matrix:
            df = pd.DataFrame(matrix)
            show_cols = [c for c in
                         ["requirement_id","requirement_text","verification_method",
                          "test_case_id","required_coverage","coverage_status"]
                         if c in df.columns]

            def _color_status(val: str):
                return {"COVERED": "background-color:#1a4731;color:#3fb950",
                        "PARTIAL": "background-color:#3d2b00;color:#f0c000",
                        "MISSING": "background-color:#3d1a1a;color:#f85149"}.get(val, "")

            st.dataframe(
                df[show_cols].style.applymap(_color_status, subset=["coverage_status"]),
                use_container_width=True, height=380, hide_index=True,
            )

            st.download_button(
                "⬇️ Export CSV",
                data=df.to_csv(index=False),
                file_name="traceability_matrix.csv",
                mime="text/csv",
            )

# ============================================================
# TAB 4 — GAP ANALYSIS
# ============================================================

with tab4:
    st.subheader("DO-178C Compliance Gap Analysis")
    st.caption(
        "Paste or upload your project documents. The agent checks coverage "
        "against every required DO-178C objective for the active DAL level."
    )

    proj_files = st.file_uploader(
        "Upload project documents (TXT/PDF)",
        type=["txt","pdf"], accept_multiple_files=True, key="gap_files",
    )
    proj_text = st.text_area(
        "Or paste combined document content:",
        height=120,
        placeholder=(
            "Software Development Plan v2.3 … The Software Verification Plan (SVP) … "
            "source code baseline … test results … MC/DC coverage achieved …"
        ),
        key="gap_txt",
    )

    if st.button("Run Gap Analysis", use_container_width=True, type="primary"):
        combined = proj_text
        for f in proj_files:
            raw = f.read()
            combined += "\n\n" + raw.decode("utf-8", errors="ignore")

        if not combined.strip():
            st.warning("Provide project documents or text.")
        else:
            with st.spinner("Analysing compliance gaps…"):
                result = _post("gap-analysis", json={
                    "session_id": SID, "project_text": combined,
                })
            if result:
                st.session_state.last_gap = result

    if st.session_state.last_gap:
        data = st.session_state.last_gap
        s = data.get("summary", {})
        risk = s.get("risk", "UNKNOWN")

        st.divider()

        # Risk + score banner
        risk_css = f"risk-{risk}"
        st.markdown(
            f'<div style="margin-bottom:0.8rem">Certification Risk: '
            f'<span class="{risk_css}">● {risk}</span> &nbsp;|&nbsp; '
            f'Compliance Score: <strong style="color:#58a6ff">{s.get("compliance_score",0)}%</strong> '
            f'&nbsp;|&nbsp; DAL-{s.get("dal_level","?")}</div>',
            unsafe_allow_html=True,
        )

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Objectives", s.get("total", 0))
        c2.metric("Covered ✅",       s.get("covered", 0))
        c3.metric("Partial ⚠️",       s.get("partial", 0))
        c4.metric("Missing ❌",        s.get("missing", 0))

        items = data.get("gap_analysis", [])
        if items:
            df_gap = pd.DataFrame(items)

            # Progress bars by section
            st.markdown("#### Coverage by Section")
            if "section" in df_gap.columns:
                for sec, grp in df_gap.groupby("section"):
                    total_sec = len(grp)
                    cov = (grp["status"] == "COVERED").sum()
                    pct = cov / total_sec if total_sec else 0
                    bar_color = "#3fb950" if pct >= 0.8 else "#f0c000" if pct >= 0.5 else "#f85149"
                    st.markdown(
                        f'<div style="margin:0.3rem 0">'
                        f'<span style="font-size:0.82rem;color:#8b949e">{sec}</span>'
                        f'<div style="background:#30363d;border-radius:4px;height:10px;margin-top:3px">'
                        f'<div style="background:{bar_color};width:{pct*100:.0f}%;height:10px;'
                        f'border-radius:4px"></div></div>'
                        f'<span style="font-size:0.75rem;color:{bar_color}">{cov}/{total_sec} covered</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Detailed Results")
            show_cols = [c for c in ["objective_id","section","requirement","status","evidence"]
                         if c in df_gap.columns]

            def _gap_color(val: str):
                return {"COVERED": "background-color:#1a4731;color:#3fb950",
                        "PARTIAL": "background-color:#3d2b00;color:#f0c000",
                        "MISSING": "background-color:#3d1a1a;color:#f85149"}.get(val, "")

            display_df = df_gap[show_cols].copy()
            if "evidence" in display_df.columns:
                display_df["evidence"] = display_df["evidence"].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )

            st.dataframe(
                display_df.style.applymap(_gap_color, subset=["status"]),
                use_container_width=True, height=420, hide_index=True,
            )

        # Critical gaps
        critical = data.get("critical_gaps", [])
        if critical:
            st.error("**Critical Gaps (immediate action required):**\n" +
                     "\n".join(f"• {g}" for g in critical))

        st.download_button(
            "⬇️ Export Gap Report (JSON)",
            data=json.dumps(data, indent=2),
            file_name="gap_analysis.json",
            mime="application/json",
        )

# ============================================================
# TAB 5 — IMPACT ANALYZER
# ============================================================

with tab5:
    st.subheader("Software Change Impact Analyzer")
    st.caption(
        "Describe a proposed change — the agent identifies affected requirements, "
        "tests, documents and assigns a risk level based on the active DAL."
    )

    change_desc = st.text_area(
        "Change description:",
        height=120,
        placeholder=(
            "Example: Modify the altitude calculation algorithm to use GPS-INS fusion "
            "instead of barometric altitude only. The new algorithm uses a Kalman filter "
            "combining GPS position data with INS measurements."
        ),
        key="impact_desc",
    )
    components_in = st.text_input(
        "Affected components (comma-separated, optional):",
        placeholder="AltitudeModule, NavigationFilter, FlightControlLaw",
        key="impact_comps",
    )

    col_dal_i, col_run = st.columns([3, 1])
    with col_dal_i:
        i_dal = st.selectbox(
            "DAL for assessment",
            ["Use session DAL", "A — Catastrophic", "B — Hazardous",
             "C — Major", "D — Minor"],
            key="impact_dal",
        )
        if i_dal != "Use session DAL":
            _set_dal(i_dal[0])
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_impact = st.button("Analyze Impact", use_container_width=True, type="primary")

    if run_impact:
        if not change_desc.strip():
            st.warning("Please describe the change.")
        else:
            with st.spinner("Analysing impact…"):
                result = _post("impact-analysis", json={
                    "session_id": SID,
                    "change_description": change_desc.strip(),
                    "affected_components": components_in,
                })
            if result:
                st.session_state.last_impact = result

    if st.session_state.last_impact:
        data = st.session_state.last_impact
        risk = data.get("risk_level", "MEDIUM")
        dal_used = data.get("dal_level", "?")
        re_verif = data.get("re_verification_required", True)

        st.divider()

        # Risk banner
        risk_css = f"risk-{risk}"
        st.markdown(
            f'<div style="font-size:1.2rem;margin-bottom:0.8rem">'
            f'Risk Level: <span class="{risk_css}">● {risk}</span> &nbsp;|&nbsp; '
            f'DAL-{dal_used} &nbsp;|&nbsp; '
            f'Re-verification: <strong>{"Required" if re_verif else "Optional"}</strong></div>',
            unsafe_allow_html=True,
        )

        # Change categories
        cats = data.get("change_categories", [])
        if cats:
            st.markdown("**Change categories:** " +
                        " · ".join(f"`{c.replace('_',' ').title()}`" for c in cats))

        # Risk gauge
        score = {"HIGH": 88, "MEDIUM": 52, "LOW": 18}.get(risk, 50)
        gauge_color = {"HIGH": "#f85149", "MEDIUM": "#f0a500", "LOW": "#3fb950"}.get(risk, "#888")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Risk Score (DAL-{dal_used})", "font": {"color": "#e6edf3", "size": 14}},
            number={"font": {"color": "#e6edf3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 33],  "color": "#1a4731"},
                    {"range": [33, 66], "color": "#3d2b00"},
                    {"range": [66, 100],"color": "#3d1a1a"},
                ],
            },
        ))
        fig_g.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3", height=220)
        st.plotly_chart(fig_g, use_container_width=True)

        # Impact cards
        imp = data.get("impact_assessment", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Affected Requirements**")
            for r in imp.get("affected_requirements", []):
                st.markdown(f"• {r}")
        with c2:
            st.markdown("**Affected Tests**")
            for t in imp.get("affected_tests", []):
                st.markdown(f"• {t}")
        with c3:
            st.markdown("**Affected Documents**")
            for d in imp.get("affected_documents", []):
                st.markdown(f"• {d}")
        with c4:
            st.markdown("**Required Reviews**")
            for rv in imp.get("required_reviews", []):
                st.markdown(f"• {rv}")

        # Recommended actions
        actions = data.get("recommended_actions", [])
        if actions:
            st.markdown("#### Recommended Actions")
            for i, a in enumerate(actions, 1):
                st.markdown(f"{i}. {a}")

        st.download_button(
            "⬇️ Export Impact Report (JSON)",
            data=json.dumps(data, indent=2),
            file_name="impact_analysis.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    '<div style="text-align:center;color:#6e7681;font-size:0.78rem;padding:0.8rem">'
    'DAL-Aware Compliance Agent v1.0.0 &nbsp;|&nbsp; '
    '<strong>Yash Verma</strong> — AI Engineer, M.Sc. Aerospace Engineering, TU Darmstadt '
    '&nbsp;|&nbsp; Airbus · Lilium'
    '</div>',
    unsafe_allow_html=True,
)
