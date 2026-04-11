"""
DAL-Aware Compliance Agent — Session Memory
============================================
Lightweight in-process session memory.
Stores DAL context and last 10 conversation turns per session.

Author: Yash Verma — AI Engineer, TU Darmstadt
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from langchain.memory import ConversationBufferWindowMemory


class ComplianceSessionMemory:
    """Per-session state: DAL level, conversation window, last citations."""

    MAX_TURNS = 10

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.dal_level: str | None = None
        self.active_standard: str | None = None
        self.last_citations: list[dict] = []
        self.query_count: int = 0
        self._created_at: float = time.time()

        self.langchain_memory = ConversationBufferWindowMemory(
            k=self.MAX_TURNS,
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
        )

    # --- DAL ---

    def set_dal(self, dal: str) -> None:
        """Set the active DAL level."""
        self.dal_level = dal.upper()

    def get_dal(self) -> str | None:
        return self.dal_level

    # --- Conversation ---

    def add_turn(self, human: str, ai: str) -> None:
        self.langchain_memory.chat_memory.add_user_message(human)
        self.langchain_memory.chat_memory.add_ai_message(ai)
        self.query_count += 1

    def get_messages(self) -> list:
        return self.langchain_memory.chat_memory.messages

    def clear(self) -> None:
        """Clear conversation but keep DAL context."""
        saved_dal = self.dal_level
        self.langchain_memory.clear()
        self.last_citations = []
        self.dal_level = saved_dal

    # --- Metadata ---

    def context_summary(self) -> str:
        parts = []
        if self.dal_level:
            parts.append(f"Active DAL: {self.dal_level}")
        if self.active_standard:
            parts.append(f"Standard: {self.active_standard}")
        if self.query_count:
            parts.append(f"Queries: {self.query_count}")
        return " | ".join(parts) if parts else "New session"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dal_level": self.dal_level,
            "active_standard": self.active_standard,
            "query_count": self.query_count,
            "session_age_sec": round(time.time() - self._created_at, 1),
        }

    @property
    def age_seconds(self) -> float:
        return time.time() - self._created_at


class SessionStore:
    """
    Global in-memory session store.
    Sessions expire after 4 hours of inactivity.
    """

    TTL = 4 * 3600
    _store: dict[str, ComplianceSessionMemory] = {}

    @classmethod
    def get_or_create(cls, session_id: str) -> ComplianceSessionMemory:
        cls._evict()
        if session_id not in cls._store:
            cls._store[session_id] = ComplianceSessionMemory(session_id)
        return cls._store[session_id]

    @classmethod
    def get(cls, session_id: str) -> ComplianceSessionMemory | None:
        return cls._store.get(session_id)

    @classmethod
    def _evict(cls) -> None:
        expired = [sid for sid, m in cls._store.items() if m.age_seconds > cls.TTL]
        for sid in expired:
            del cls._store[sid]

    @classmethod
    def count(cls) -> int:
        return len(cls._store)
