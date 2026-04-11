---
title: DAL-Aware Compliance Agent
emoji: ✈️
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
---

# ✈️ DAL-Aware Compliance Agent

> **Industrial-grade agentic AI for DO-178C aerospace software certification**  
> Supports DO-178C · DO-254 · ARP4761 · ARP4754A · DO-160

**Author:** Yash Verma — AI Engineer, M.Sc. Aerospace Engineering, TU Darmstadt  
Experience at Airbus and Lilium

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/YashVermaTech/dal-compliance-agent)
[![Render Backend](https://img.shields.io/badge/Backend-Render.com-purple)](https://dal-compliance-backend.onrender.com)

---

## What This Does

The DAL-Aware Compliance Agent automates the most time-consuming parts of DO-178C certification work:

| Feature | Description |
|---------|-------------|
| **Compliance Q&A** | Ask questions, get answers with section/page citations |
| **DAL-Aware Filtering** | All responses filtered to your project's DAL level (A/B/C/D) |
| **Traceability Matrix** | Maps requirements → tests → verification evidence |
| **Gap Analysis** | Compares your project docs against the DO-178C checklist |
| **Impact Analysis** | Assess risk of software changes before you make them |
| **Multi-Standard** | Routes to DO-178C, DO-254, ARP4761, ARP4754A, DO-160 |
| **Bilingual** | English + German (Deutsch) fully supported |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│      HuggingFace Spaces (Streamlit)  — Frontend          │
│  Tab 1: Ingest  Tab 2: Chat  Tab 3: Traceability         │
│  Tab 4: Gap Analysis        Tab 5: Impact Analyzer        │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────┐
│          Render.com (FastAPI)  — Backend                 │
│  POST /ingest   POST /query   POST /traceability         │
│  POST /gap-analysis   POST /impact-analysis              │
└──────────────┬────────────────────────┬─────────────────┘
               │                        │
┌──────────────▼──────────┐  ┌──────────▼─────────────────┐
│   Pinecone (free tier)  │  │  Groq — llama3-70b-8192    │
│   Semantic search       │  │  LangChain agent + 6 tools  │
│   DAL-level metadata    │  │  Conversation memory        │
└─────────────────────────┘  └────────────────────────────┘
```

---

## Setup

### 1. Set environment variables

**HuggingFace Spaces → Settings → Repository secrets:**
```
BACKEND_URL = https://your-render-app.onrender.com
```

**Render.com → Environment:**
```
GROQ_API_KEY     = your_groq_key
PINECONE_API_KEY = your_pinecone_key
PINECONE_ENV     = your_pinecone_env
```

### 2. Get free API keys

**Groq** (LLM — llama3-70b-8192, free):
- Sign up at https://console.groq.com → API Keys

**Pinecone** (Vector DB, free tier):
- Sign up at https://www.pinecone.io → Create index

### 3. Deploy backend to Render.com

```bash
# Connect your GitHub repo at render.com/new
# Render auto-reads render.yaml
```

---

## Example Queries

**English:**
- *"What MC/DC coverage is required for DAL-A software?"*
- *"What independence is required for SQA in DAL-A?"*
- *"List all required documents for DO-178C DAL-B certification"*

**Deutsch:**
- *"Welche Anforderungen gelten für DAL-B Softwareverifikation?"*
- *"Erkläre den Unterschied zwischen HLR und LLR in DO-178C"*
- *"Welche MC/DC-Abdeckung wird für DAL-A benötigt?"*

---

## DO-178C DAL Reference

| DAL | Failure Condition | Required Objectives | Coverage |
|-----|-------------------|---------------------|----------|
| A | Catastrophic | 71 | MC/DC |
| B | Hazardous | 69 | Decision |
| C | Major | 62 | Statement |
| D | Minor | 26 | Statement |

---

*This tool assists human engineers — final certification decisions require qualified DER review.*  
*Dieses System unterstützt Ingenieure — Zertifizierungsentscheidungen erfordern DER-Überprüfung.*
