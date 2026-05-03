"""
DAL-Aware Compliance Agent — FastAPI Backend (Render.com)
==========================================================
8 endpoints for the cloud-deployed compliance agent.
Reads all secrets from environment variables — no hardcoded keys.

Endpoints:
  POST /api/v1/ingest           Upload + index PDF
  POST /api/v1/set-dal          Set DAL level for session
  POST /api/v1/query            Ask compliance question
  POST /api/v1/traceability     Generate traceability matrix
  POST /api/v1/gap-analysis     Run compliance gap analysis
  POST /api/v1/impact-analysis  Run change impact analysis
  GET  /api/v1/documents        List indexed documents
  GET  /api/v1/health           Health check

Author: Yash Verma — AI Engineer, TU Darmstadt
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import PyPDF2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== DAL-Aware Compliance Agent API starting ===")
    yield
    log.info("=== API shutting down ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DAL-Aware Compliance Agent API",
    description=(
        "Aerospace software certification assistant for DO-178C, DO-254, "
        "ARP4761, ARP4754A, DO-160. Powered by Groq + Pinecone + LangChain.\n\n"
        "**Author:** Yash Verma — AI Engineer, M.Sc. Aerospace Engineering, TU Darmstadt"
    ),
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = int((time.perf_counter() - t0) * 1000)
    log.info("%s %s %d %dms", request.method, request.url.path, response.status_code, ms)
    response.headers["X-Response-Time-Ms"] = str(ms)
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SetDALRequest(BaseModel):
    session_id: str
    dal_level: str = Field(pattern="^[AaBbCcDd]$")

    @field_validator("dal_level")
    @classmethod
    def upper_dal(cls, v: str) -> str:
        return v.upper()


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=5, max_length=2000)
    dal_level: Optional[str] = None
    standard: Optional[str] = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class TraceabilityRequest(BaseModel):
    session_id: str
    requirements_text: str = Field(min_length=10, max_length=30000)
    document_name: str = "requirements.txt"


class GapAnalysisRequest(BaseModel):
    session_id: str
    project_text: str = Field(min_length=10, max_length=80000)


class ImpactAnalysisRequest(BaseModel):
    session_id: str
    change_description: str = Field(min_length=10, max_length=3000)
    affected_components: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf(content: bytes) -> list[tuple[int, str]]:
    import io
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _detect_standard(filename: str) -> str:
    patterns = {
        "DO-178C":  re.compile(r"do[-_]?178c?", re.I),
        "DO-254":   re.compile(r"do[-_]?254",   re.I),
        "DO-160":   re.compile(r"do[-_]?160",   re.I),
        "ARP4761":  re.compile(r"arp[-_]?4761",  re.I),
        "ARP4754A": re.compile(r"arp[-_]?4754a?",re.I),
    }
    for std, pat in patterns.items():
        if pat.search(filename):
            return std
    return "UNKNOWN"


def _extract_citations(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"\[(\d+)\]\s+([^—\n]+)\s+—\s+Section\s+([^,\n]+),\s+Page\s+(\d+)")
    return [
        {"index": m.group(1), "document": m.group(2).strip(),
         "section": m.group(3).strip(), "page": m.group(4)}
        for m in pattern.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

PREFIX = "/api/v1"


# --- Health ---

@app.get(f"{PREFIX}/health", tags=["System"])
async def health() -> dict:
    components: dict[str, Any] = {}

    try:
        from backend.vector_store.pinecone_client import PineconeClient
        pc = PineconeClient.get_instance()
        stats = pc.list_documents()
        components["pinecone"] = {
            "status": "ok" if pc.is_ready else "degraded",
            "vectors": stats.get("total_vector_count", 0),
        }
    except Exception as exc:
        components["pinecone"] = {"status": "error", "detail": str(exc)[:100]}

    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        models = client.models.list()
        groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        model_ids = [m.id for m in models.data]
        components["groq"] = {
            "status": "ok" if groq_model in model_ids else "model_unavailable",
            "model": groq_model,
        }
    except Exception as exc:
        components["groq"] = {"status": "error", "detail": str(exc)[:100]}

    overall = "ok" if all(v.get("status") == "ok" for v in components.values()) else "degraded"
    return {"status": overall, "components": components, "version": "1.0.0"}


# --- Ingest ---

@app.post(f"{PREFIX}/ingest", tags=["Ingestion"])
async def ingest_document(
    file: UploadFile = File(...),
    dal_level: str = Form(default="ALL"),
    session_id: str = Form(default=""),
) -> dict:
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use PDF or TXT.")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 50 MB limit.")

    try:
        if suffix == ".pdf":
            pages = _extract_pdf(content)
        else:
            text = content.decode("utf-8", errors="ignore")
            pages = [(1, text)]

        if not pages:
            raise HTTPException(422, "No readable text found in file.")

        standard = _detect_standard(file.filename)

        from backend.vector_store.pinecone_client import PineconeClient
        pc = PineconeClient.get_instance()

        chunks = pc.embed_and_chunk(
            pages=pages,
            filename=file.filename,
            standard=standard,
            dal_override=dal_level,
        )

        upserted = pc.upsert_chunks(chunks)

        log.info("Ingested %s → %d chunks (%s)", file.filename, upserted, standard)
        return {
            "filename": file.filename,
            "status": "indexed",
            "standard": standard,
            "chunks_indexed": upserted,
            "message": f"Successfully indexed {upserted} chunks from {file.filename}",
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Ingest failed for %s: %s", file.filename, exc, exc_info=True)
        raise HTTPException(500, f"Ingestion failed: {str(exc)[:300]}")


# --- Documents ---

@app.get(f"{PREFIX}/documents", tags=["Ingestion"])
async def list_documents() -> dict:
    try:
        from backend.vector_store.pinecone_client import PineconeClient
        pc = PineconeClient.get_instance()
        stats = pc.list_documents()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "index": os.environ.get("PINECONE_INDEX_NAME", "dal-compliance-index"),
            "status": "ok" if pc.is_ready else "degraded",
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


# --- Set DAL ---

@app.post(f"{PREFIX}/set-dal", tags=["Session"])
async def set_dal(request: SetDALRequest) -> dict:
    try:
        from backend.agent.memory import SessionStore
        mem = SessionStore.get_or_create(request.session_id)
        mem.set_dal(request.dal_level)

        descriptions = {
            "A": "Catastrophic — 71 objectives — MC/DC coverage required",
            "B": "Hazardous — 69 objectives — Decision coverage required",
            "C": "Major — 62 objectives — Statement coverage required",
            "D": "Minor — 26 objectives — Statement coverage required",
        }
        return {
            "session_id": request.session_id,
            "dal_level": request.dal_level,
            "description": descriptions[request.dal_level],
            "message": f"DAL-{request.dal_level} set for session.",
        }
    except Exception as exc:
        log.error("set-dal failed: %s", exc)
        raise HTTPException(500, str(exc))


# --- Query ---

@app.post(f"{PREFIX}/query", tags=["Agent"])
async def query_compliance(request: QueryRequest) -> dict:
    """
    Ask a compliance question. Returns answer + citations + DAL context.
    Supports English and German.
    Direct RAG: Pinecone search -> Groq chat completion. No LangChain agent.
    """
    try:
        from backend.vector_store.pinecone_client import PineconeClient
        from backend.agent.memory import SessionStore
        from groq import Groq

        # Session memory
        mem = SessionStore.get_or_create(request.session_id)
        if request.dal_level:
            mem.set_dal(request.dal_level.upper())
        dal = mem.get_dal() or "B"

        t0 = time.perf_counter()

        # Step 1: Search Pinecone directly
        pc = PineconeClient.get_instance()
        try:
            results = pc.semantic_search(
                query=request.question,
                dal_level=dal,
                standard=request.standard,
                top_k=5,
            )
        except Exception as search_exc:
            log.warning("Pinecone search failed, continuing without context: %s", search_exc)
            results = []

        # Step 2: Build context string
        if results:
            context_lines = [f"Relevant compliance documents [DAL-{dal}]:\n"]
            for i, r in enumerate(results, 1):
                context_lines.append(
                    f"[Source {i}] {r['standard']} — Section {r['section_number']}: "
                    f"{r['section_title']}\n"
                    f"Document: {r['filename']} | Page: {r['page_number']}\n"
                    f"{r['text'][:600]}\n"
                )
            context_lines.append("\nCITATIONS:")
            for i, r in enumerate(results, 1):
                context_lines.append(
                    f"[{i}] {r['filename']} — Section {r['section_number']}, "
                    f"Page {r['page_number']} (relevance: {r['score']:.2f})"
                )
            context = "\n".join(context_lines)
        else:
            context = (
                "No compliance documents are currently indexed. "
                "Please upload DO-178C or other standard PDFs via the Document Ingestion tab first."
            )

        # Step 3: Send context + question to Groq as plain chat completion
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        chat = groq_client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the DAL-Aware Compliance Agent — an expert aerospace software "
                        "certification assistant for DO-178C, DO-254, ARP4761, ARP4754A, and DO-160. "
                        f"The active Design Assurance Level is DAL-{dal}. "
                        "Answer questions using only the context provided below. "
                        "Always cite the section number and page number from the sources. "
                        "If no context is available, tell the user to upload compliance documents first. "
                        "Respond in the same language as the question (English or German). "
                        "Final certification decisions require qualified DER review."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {request.question}",
                },
            ],
        )

        answer = chat.choices[0].message.content
        latency = int((time.perf_counter() - t0) * 1000)

        mem.add_turn(request.question, answer)

        citations = _extract_citations(context)
        confidence = min(0.95, 0.5 + len(citations) * 0.1)

        log.info("Query done | session=%s dal=%s | %dms | %d sources",
                 request.session_id, dal, latency, len(results))

        return {
            "session_id": request.session_id,
            "answer": answer,
            "citations": citations,
            "dal_level": dal,
            "confidence_score": round(confidence, 2),
            "latency_ms": latency,
            "tool_calls_count": 1,
        }

    except Exception as exc:
        log.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Query failed: {str(exc)[:300]}")


# --- Traceability ---

@app.post(f"{PREFIX}/traceability", tags=["Agent"])
async def traceability(request: TraceabilityRequest) -> dict:
    try:
        from backend.agent.tools import GenerateTraceabilityMatrixTool
        from backend.agent.memory import SessionStore

        mem = SessionStore.get_or_create(request.session_id)
        tool = GenerateTraceabilityMatrixTool(session_state={"dal_level": mem.get_dal() or "B"})

        t0 = time.perf_counter()
        raw = tool._run(request.requirements_text, request.document_name)
        latency = int((time.perf_counter() - t0) * 1000)

        data = json.loads(raw)
        data["latency_ms"] = latency
        data["session_id"] = request.session_id
        return data

    except Exception as exc:
        log.error("Traceability failed: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc)[:300])


# --- Gap Analysis ---

@app.post(f"{PREFIX}/gap-analysis", tags=["Agent"])
async def gap_analysis(request: GapAnalysisRequest) -> dict:
    try:
        from backend.agent.tools import DetectComplianceGapsTool
        from backend.agent.memory import SessionStore

        mem = SessionStore.get_or_create(request.session_id)
        tool = DetectComplianceGapsTool(session_state={"dal_level": mem.get_dal() or "B"})

        t0 = time.perf_counter()
        raw = tool._run(request.project_text)
        latency = int((time.perf_counter() - t0) * 1000)

        data = json.loads(raw)
        data["latency_ms"] = latency
        data["session_id"] = request.session_id
        return data

    except Exception as exc:
        log.error("Gap analysis failed: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc)[:300])


# --- Impact Analysis ---

@app.post(f"{PREFIX}/impact-analysis", tags=["Agent"])
async def impact_analysis(request: ImpactAnalysisRequest) -> dict:
    try:
        from backend.agent.tools import DALImpactAnalyzerTool
        from backend.agent.memory import SessionStore

        mem = SessionStore.get_or_create(request.session_id)
        tool = DALImpactAnalyzerTool(session_state={"dal_level": mem.get_dal() or "B"})

        t0 = time.perf_counter()
        raw = tool._run(request.change_description, request.affected_components)
        latency = int((time.perf_counter() - t0) * 1000)

        data = json.loads(raw)
        data["latency_ms"] = latency
        data["session_id"] = request.session_id
        return data

    except Exception as exc:
        log.error("Impact analysis failed: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc)[:300])


# --- Root ---

@app.get("/", tags=["System"])
async def root() -> dict:
    return {
        "name": "DAL-Aware Compliance Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "standards": ["DO-178C", "DO-254", "DO-160", "ARP4761", "ARP4754A"],
        "author": "Yash Verma — AI Engineer, TU Darmstadt",
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
