# DAL-Aware Compliance Agent — System Architecture

## Deployment Topology

```
┌──────────────────────────────────────────────────────────────┐
│           HuggingFace Spaces (Streamlit)                      │
│                 app.py — 5 Tabs                               │
│  Tab 1: Ingest · Tab 2: Chat · Tab 3: Traceability           │
│  Tab 4: Gap Analysis · Tab 5: Impact Analyzer                 │
│                                                              │
│  Secret: BACKEND_URL → https://dal-compliance-backend.onrender.com │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTPS REST
┌──────────────────────────▼───────────────────────────────────┐
│              Render.com — FastAPI (backend/main.py)           │
│                                                              │
│  POST /api/v1/ingest         POST /api/v1/set-dal            │
│  POST /api/v1/query          POST /api/v1/traceability        │
│  POST /api/v1/gap-analysis   POST /api/v1/impact-analysis     │
│  GET  /api/v1/documents      GET  /api/v1/health              │
│                                                              │
│  Env vars: GROQ_API_KEY · PINECONE_API_KEY · PINECONE_ENV    │
└──────────────┬──────────────────────────┬────────────────────┘
               │                          │
┌──────────────▼──────────┐  ┌────────────▼──────────────────┐
│  Pinecone (free tier)   │  │  Groq — llama3-70b-8192       │
│  Serverless index       │  │  Free tier: 30 req/min        │
│  384-dim cosine         │  │  LangChain agent + 6 tools    │
│  DAL metadata filter    │  │  Conversation memory          │
└─────────────────────────┘  └────────────────────────────────┘
         ▲
         │ embed + upsert (on ingest)
┌────────┴────────────────────────────────────────────────────┐
│  HuggingFace sentence-transformers/all-MiniLM-L6-v2         │
│  Free, no API key, runs on Render CPU                       │
└─────────────────────────────────────────────────────────────┘
```

## Request Flow

### Compliance Query
```
User (Streamlit Tab 2)
  → POST /api/v1/query  {session_id, question, dal_level}
  → get_agent(session_id)
  → LangChain AgentExecutor
      → multi_standard_router (detect standard)
      → query_compliance → PineconeClient.semantic_search (DAL-filtered)
      → Groq LLM generates answer + cites sources
  ← {answer, citations, dal_level, confidence_score, latency_ms}
```

### Document Ingestion
```
User (Streamlit Tab 1) uploads PDF
  → POST /api/v1/ingest  {file, dal_level}
  → PyPDF2 extracts text per page
  → PineconeClient.embed_and_chunk()
      → regex section splitting
      → SentenceTransformer.encode() (384-dim)
      → DAL auto-detection per chunk
  → PineconeClient.upsert_chunks() → Pinecone index
  ← {chunks_indexed, standard, status}
```

### Gap Analysis
```
User pastes project docs (Streamlit Tab 4)
  → POST /api/v1/gap-analysis  {session_id, project_text}
  → DetectComplianceGapsTool._run()
      → load DAL_OBJECTIVES[dal] (26–71 required items)
      → keyword matching per objective
      → COVERED / PARTIAL / MISSING classification
  ← {gap_analysis[], summary{score, risk}, critical_gaps[]}
```

## Agent Tools

| # | Tool | Purpose |
|---|------|---------|
| 1 | `set_dal_level` | Set DAL (A/B/C/D) for session, persist in memory |
| 2 | `query_compliance` | Pinecone semantic search → Groq synthesis → citations |
| 3 | `generate_traceability_matrix` | REQ → TEST → VERIFY mapping |
| 4 | `detect_compliance_gaps` | DO-178C checklist coverage check |
| 5 | `dal_impact_analyzer` | Change categorization → risk + re-verification list |
| 6 | `multi_standard_router` | Keyword-based standard detection (DO-178C/DO-254/ARP4761…) |

## Data Models

### Pinecone Vector Record
```json
{
  "id": "md5_chunk_id",
  "values": [0.12, -0.03, ...],  // 384-dim
  "metadata": {
    "text": "Section 6.4 ...",
    "filename": "DO-178C.pdf",
    "standard": "DO-178C",
    "dal_level": "A",
    "section_number": "6.4",
    "section_title": "Test Cases and Procedures",
    "page_number": 45
  }
}
```

### Session Memory
```python
ComplianceSessionMemory:
  session_id: str
  dal_level: str | None       # "A" / "B" / "C" / "D"
  active_standard: str | None
  langchain_memory: ConversationBufferWindowMemory(k=10)
  query_count: int
```

## Environment Variables

| Variable | Where | Description |
|----------|-------|-------------|
| `GROQ_API_KEY` | Render | Groq API key (free at console.groq.com) |
| `PINECONE_API_KEY` | Render | Pinecone API key (free tier) |
| `PINECONE_ENV` | Render | Pinecone environment (e.g. `us-east-1`) |
| `BACKEND_URL` | HuggingFace secret | URL of the Render backend |
| `PINECONE_INDEX_NAME` | Render | Index name (default: `dal-compliance-index`) |
| `GROQ_MODEL` | Render | Model ID (default: `llama3-70b-8192`) |

## DO-178C DAL Objectives

| DAL | Failure | Objectives | Coverage |
|-----|---------|-----------|---------|
| A | Catastrophic | 71 | MC/DC |
| B | Hazardous | 69 | Decision |
| C | Major | 62 | Statement |
| D | Minor | 26 | Statement |
