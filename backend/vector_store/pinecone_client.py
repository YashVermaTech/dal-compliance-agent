"""
DAL-Aware Compliance Agent — Pinecone Vector Store Client
==========================================================
Wraps Pinecone free-tier serverless index for DAL-filtered
semantic search over aerospace compliance documents.

Author: Yash Verma — AI Engineer, TU Darmstadt
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "dal-compliance-index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


class PineconeClient:
    """
    Singleton Pinecone client with lazy initialization.

    Provides:
    - DAL-filtered semantic search
    - Chunk upsert (used by ingest endpoint)
    - Document listing
    """

    _instance: PineconeClient | None = None
    _model = None  # SentenceTransformer singleton

    def __init__(self) -> None:
        self._index = None
        self._ready = False

    @classmethod
    def get_instance(cls) -> "PineconeClient":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._init()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init(self) -> None:
        """Connect to Pinecone and load embedding model."""
        if not PINECONE_API_KEY:
            log.warning("PINECONE_API_KEY not set — vector search unavailable.")
            return
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = [idx.name for idx in pc.list_indexes()]

            if PINECONE_INDEX not in existing:
                log.info("Creating Pinecone index: %s", PINECONE_INDEX)
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self._index = pc.Index(PINECONE_INDEX)
            self._load_model()
            self._ready = True
            log.info("Pinecone client ready (index: %s)", PINECONE_INDEX)

        except Exception as exc:
            log.error("Pinecone init failed: %s", exc, exc_info=True)
            self._ready = False

    def _load_model(self) -> None:
        if PineconeClient._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model: %s", EMBEDDING_MODEL)
            PineconeClient._model = SentenceTransformer(EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return a normalized embedding for the given text."""
        self._load_model()
        vec = PineconeClient._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def semantic_search(
        self,
        query: str,
        dal_level: str | None = None,
        standard: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Cosine-similarity search with optional DAL + standard metadata filter.

        Args:
            query:     Natural language query.
            dal_level: Filter to this DAL (or 'ALL').
            standard:  Filter to this standard (e.g. 'DO-178C').
            top_k:     Max results to return.

        Returns:
            List of result dicts: text, score, filename, section_number, page_number …
        """
        if not self._ready:
            log.warning("Pinecone not ready — returning empty results.")
            return []

        vector = self._embed(query)

        meta_filter: dict[str, Any] = {}
        if dal_level and dal_level not in ("ALL", ""):
            meta_filter["dal_level"] = {"$in": [dal_level, "ALL"]}
        if standard:
            meta_filter["standard"] = {"$eq": standard}

        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        if meta_filter:
            kwargs["filter"] = meta_filter

        resp = self._index.query(**kwargs)

        return [
            {
                "chunk_id":      m["id"],
                "score":         float(m["score"]),
                "text":          m["metadata"].get("text", ""),
                "filename":      m["metadata"].get("filename", ""),
                "standard":      m["metadata"].get("standard", ""),
                "dal_level":     m["metadata"].get("dal_level", ""),
                "section_number": m["metadata"].get("section_number", ""),
                "section_title": m["metadata"].get("section_title", ""),
                "page_number":   int(m["metadata"].get("page_number", 0)),
            }
            for m in resp.get("matches", [])
        ]

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """
        Upsert pre-embedded chunks into Pinecone.

        Each chunk must have: chunk_id, text, embedding (list[float]),
        filename, standard, dal_level, section_number, section_title, page_number.

        Returns number of vectors upserted.
        """
        if not self._ready or not chunks:
            return 0

        batch_size = 100
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            vectors = [
                {
                    "id": c["chunk_id"],
                    "values": c["embedding"],
                    "metadata": {
                        "text":           c["text"][:1800],
                        "filename":       c.get("filename", ""),
                        "standard":       c.get("standard", ""),
                        "dal_level":      c.get("dal_level", "ALL"),
                        "section_number": c.get("section_number", "N/A"),
                        "section_title":  c.get("section_title", ""),
                        "page_number":    int(c.get("page_number", 1)),
                    },
                }
                for c in batch
            ]
            self._index.upsert(vectors=vectors)
            total += len(vectors)

        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def embed_and_chunk(
        self,
        pages: list[tuple[int, str]],
        filename: str,
        standard: str,
        dal_override: str = "ALL",
    ) -> list[dict[str, Any]]:
        """
        Chunk page texts into sections, generate embeddings, and return
        records ready for upsert.

        Args:
            pages:        List of (page_number, text) tuples.
            filename:     Source document filename.
            standard:     Standard name (e.g. 'DO-178C').
            dal_override: DAL level to tag all chunks with, or 'ALL' for auto-detect.
        """
        import re

        self._load_model()

        # Section-aware splitting
        full_text = "\n".join(t for _, t in pages)
        sec_re = re.compile(r"(?m)^((?:\d+\.)+\d*)\s+([A-Z][^\n]{3,80})\n")
        matches = list(sec_re.finditer(full_text))

        raw: list[dict] = []
        if matches:
            for idx, m in enumerate(matches):
                sec_num = m.group(1)
                sec_title = m.group(2).strip()
                start = m.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
                text = full_text[start:end].strip()
                if text:
                    raw.append({"section_number": sec_num, "section_title": sec_title,
                                "text": text})
        else:
            # Fallback: paragraph chunks
            paras = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 50]
            for i, p in enumerate(paras):
                raw.append({"section_number": "N/A", "section_title": "General",
                            "text": p[:1500]})

        if not raw:
            return []

        # Generate embeddings in one batch
        texts = [r["text"][:512] for r in raw]
        embeddings = PineconeClient._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()

        # Auto-detect DAL from text
        dal_kws = {
            "A": ["level a", "dal-a", "catastrophic"],
            "B": ["level b", "dal-b", "hazardous"],
            "C": ["level c", "dal-c", "major"],
            "D": ["level d", "dal-d", "minor"],
        }

        chunks: list[dict[str, Any]] = []
        for r, emb in zip(raw, embeddings):
            if dal_override != "ALL":
                dal = dal_override
            else:
                tl = r["text"].lower()
                dal = next(
                    (d for d, kws in dal_kws.items() if any(k in tl for k in kws)),
                    "ALL",
                )

            # Best page heuristic
            page = 1
            for pnum, ptext in pages:
                if r["text"][:80].replace("\n", " ") in ptext.replace("\n", " "):
                    page = pnum
                    break

            cid = hashlib.md5(
                f"{filename}_{r['section_number']}_{r['text'][:40]}".encode()
            ).hexdigest()

            chunks.append({
                "chunk_id":       cid,
                "text":           r["text"],
                "filename":       filename,
                "standard":       standard,
                "dal_level":      dal,
                "section_number": r["section_number"],
                "section_title":  r["section_title"],
                "page_number":    page,
                "embedding":      emb,
            })

        return chunks

    def list_documents(self) -> dict[str, Any]:
        """Return index stats from Pinecone."""
        if not self._ready:
            return {"total_vector_count": 0}
        try:
            return dict(self._index.describe_index_stats())
        except Exception as exc:
            log.error("Failed to fetch Pinecone stats: %s", exc)
            return {"total_vector_count": 0}

    @property
    def is_ready(self) -> bool:
        return self._ready
