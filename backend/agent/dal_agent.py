"""
DAL-Aware Compliance Agent — LangChain Agent Orchestrator
==========================================================
Multi-step agent using Groq (llama-3.3-70b-versatile) + 6 compliance tools.
Supports English and German. Session memory persists across requests.

Author: Yash Verma — AI Engineer, TU Darmstadt
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from backend.agent.memory import SessionStore
from backend.agent.tools import build_tools

log = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are the DAL-Aware Compliance Agent — an expert aerospace software \
certification assistant for DO-178C, DO-254, ARP4761, ARP4754A, and DO-160.

Your users are aerospace engineers (Airbus, Boeing, Lilium, DLR) with advanced \
certification knowledge. Be precise, cite sources, and apply the active DAL level.

Session context: {session_context}

Rules:
- Always cite document, section, and page number when answering from standards
- Apply DAL-specific filtering (DAL-A: 71 objectives / MC/DC; DAL-D: 26 / Statement)
- Respond in the same language as the question (English or German / Deutsch)
- Never guess requirements — use the query_compliance tool first
- For ambiguous standard questions, use multi_standard_router first
- Format complex answers with bullet points or numbered lists
- When calling tools, always use valid JSON with double quotes

Safety: This system assists engineers. Final certification decisions require qualified \
DER (Designated Engineering Representative) review.
Sicherheitshinweis: Zertifizierungsentscheidungen erfordern qualifizierte DER-Überprüfung."""


def _build_prompt(session_context: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(session_context=session_context)),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])


class DALComplianceAgent:
    """
    Per-session compliance agent.
    Lazy-initializes the LangChain executor on first use.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._memory = SessionStore.get_or_create(session_id)
        self._session_state: dict[str, Any] = {
            "dal_level": self._memory.get_dal() or "",
            "session_id": session_id,
        }
        self._executor: AgentExecutor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> dict[str, Any]:
        """
        Execute a compliance query and return a structured response.

        Returns:
            dict with keys: answer, citations, dal_level, tool_calls, latency_ms
        """
        t0 = time.perf_counter()
        executor = self._get_executor()

        # Sync DAL from memory in case it was updated externally
        if self._memory.get_dal():
            self._session_state["dal_level"] = self._memory.get_dal()

        log.info("Agent query | session=%s dal=%s | %.80s",
                 self.session_id, self._session_state.get("dal_level"), user_input)

        try:
            result = executor.invoke({
                "input": user_input,
                "chat_history": self._memory.get_messages(),
            })
        except Exception as exc:
            log.error("Agent execution error: %s", exc, exc_info=True)
            return {
                "answer": f"Error processing request: {str(exc)[:300]}",
                "citations": [],
                "dal_level": self._session_state.get("dal_level"),
                "tool_calls": [],
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": str(exc),
            }

        answer = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        tool_calls = [
            {
                "tool": action.tool,
                "input": str(action.tool_input)[:200],
                "output_preview": str(obs)[:200],
            }
            for action, obs in steps
        ]

        citations = []
        for action, obs in steps:
            if action.tool == "query_compliance":
                citations.extend(_extract_citations(str(obs)))

        # Sync DAL back to memory
        if self._session_state.get("dal_level"):
            self._memory.set_dal(self._session_state["dal_level"])

        self._memory.add_turn(user_input, answer)

        latency = int((time.perf_counter() - t0) * 1000)
        log.info("Agent done | session=%s | %dms | %d tool calls",
                 self.session_id, latency, len(tool_calls))

        return {
            "answer": answer,
            "citations": citations,
            "dal_level": self._session_state.get("dal_level") or self._memory.get_dal(),
            "tool_calls": tool_calls,
            "latency_ms": latency,
        }

    def set_dal(self, dal: str) -> None:
        """Directly set the DAL level without running the agent."""
        self._session_state["dal_level"] = dal.upper()
        self._memory.set_dal(dal)
        self._executor = None  # Rebuild prompt with updated context

    def clear_history(self) -> None:
        self._memory.clear()
        self._executor = None

    def session_info(self) -> dict[str, Any]:
        return self._memory.to_dict()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_executor(self) -> AgentExecutor:
        if self._executor is not None:
            return self._executor

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,        # 0 = deterministic, reduces malformed tool call JSON
            max_tokens=4096,
            timeout=60,
        )

        tools = build_tools(self._session_state)
        prompt = _build_prompt(self._memory.context_summary())

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self._memory.langchain_memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=6,
            return_intermediate_steps=True,
        )

        log.info("Executor built for session %s | model=%s", self.session_id, GROQ_MODEL)
        return self._executor


# ------------------------------------------------------------------
# Module-level agent cache (one per session_id)
# ------------------------------------------------------------------

_cache: dict[str, DALComplianceAgent] = {}


def get_agent(session_id: str) -> DALComplianceAgent:
    """Get or create the agent for a session."""
    if session_id not in _cache:
        _cache[session_id] = DALComplianceAgent(session_id)
    return _cache[session_id]


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _extract_citations(obs_text: str) -> list[dict[str, str]]:
    """Parse citation lines from query_compliance tool output."""
    pattern = re.compile(r"\[(\d+)\]\s+([^—\n]+)\s+—\s+Section\s+([^,\n]+),\s+Page\s+(\d+)")
    return [
        {"index": m.group(1), "document": m.group(2).strip(),
         "section": m.group(3).strip(), "page": m.group(4)}
        for m in pattern.finditer(obs_text)
    ]
