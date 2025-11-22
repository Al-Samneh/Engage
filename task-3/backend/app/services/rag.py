from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

from ..config import Settings
from ..schemas import (
    AppliedFilters,
    DocumentSnippet,
    RestaurantSearchPayload,
    RestaurantSearchRequest,
)
from .chat_store import ChatStore


def _ensure_task1_on_path(task1_dir: Path) -> None:
    """
    Task 1 was built as a standalone project. We load its .env and add both the root and src
    directories to sys.path so imports (config, agents, etc.) resolve at runtime.
    """
    base_dir = task1_dir.parent  # Go up to repo root (Engage/)
    env_locations = [
        task1_dir / ".env",  # task-1/.env
        base_dir / "engagetest" / ".env",  # engagetest/.env
        base_dir / ".env",  # root/.env
        Path(".env"),  # current directory
    ]

    env_loaded = False
    for env_file in env_locations:
        if env_file.exists():
            load_dotenv(env_file, override=True)
            logger.info(f"Loaded .env from {env_file}")
            env_loaded = True
            break

    if not env_loaded:
        load_dotenv()
        logger.warning(
            f"No .env file found in expected locations: {[str(p) for p in env_locations]}"
        )

    path_str = str(task1_dir)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
    src_path = str(task1_dir / "src")
    if src_path in sys.path:
        sys.path.remove(src_path)
    sys.path.insert(0, src_path)

    for module_name in list(sys.modules.keys()):
        if module_name in {"config", "state", "agents", "ingest"}:
            module = sys.modules[module_name]
            module_file = getattr(module, "__file__", "") or ""
            if module_file and str(task1_dir) not in module_file:
                sys.modules.pop(module_name, None)


@dataclass
class RAGResult:
    payload: RestaurantSearchPayload
    latency_ms: int
    trace_id: uuid.UUID


class RAGService:
    """Keeps an in-memory LangGraph instance and mediates chat history storage."""

    def __init__(self, settings: Settings, chat_store: ChatStore) -> None:
        _ensure_task1_on_path(settings.task1_dir)

        # Fix the database path to be absolute (relative paths break when running from API)
        import config as task1_config  # type: ignore

        db_path_absolute = str(settings.task1_dir / "db")
        task1_config.DB_PATH = db_path_absolute
        task1_config.get_chroma_client.cache_clear()
        logger.info(f"Set ChromaDB path to: {db_path_absolute}")

        from main import build_graph  # type: ignore

        self._graph = build_graph()
        self._chat_store = chat_store
        self._settings = settings

    def _prepare_history(self, conversation_id: Optional[uuid.UUID]) -> List[str]:
        """Fetch last few turns for multi-turn conversations."""
        if not conversation_id:
            return []
        return self._chat_store.fetch(conversation_id)

    def _persist_turn(
        self, conversation_id: Optional[uuid.UUID], user_input: str, ai_output: str
    ) -> None:
        """Store turns so future queries inherit context."""
        if not conversation_id:
            return
        self._chat_store.append(conversation_id, f"User: {user_input}", f"AI: {ai_output}")

    def search(self, request: RestaurantSearchRequest) -> RAGResult:
        """Invoke LangGraph and translate results into API schemas."""
        trace_id = uuid.uuid4()
        start = time.perf_counter()
        history = self._prepare_history(request.conversation_id)
        inputs = {"question": request.question, "messages": history}

        try:
            result = self._graph.invoke(inputs)
        except Exception as exc:  # pragma: no cover - network/LLM errors
            logger.exception(
                f"RAG execution failed for question: {request.question[:100]}",
                exc_info=exc,
            )
            error_msg = str(exc)
            if "GOOGLE_API_KEY" in error_msg or "api_key" in error_msg.lower():
                error_msg = (
                    "Google API key not configured. Please set GOOGLE_API_KEY environment variable."
                )
            elif "chroma" in error_msg.lower() or "database" in error_msg.lower():
                error_msg = "Database connection error. Please check ChromaDB setup."

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "RAG_EXECUTION_FAILED",
                    "message": f"Unable to generate recommendations: {error_msg}",
                    "trace_id": str(trace_id),
                },
            ) from exc

        answer = result.get("generation", "")
        documents = result.get("documents", [])
        filters = result.get("filters", {})

        payload = RestaurantSearchPayload(
            answer=answer,
            applied_filters=AppliedFilters(**filters),
            documents=[
                DocumentSnippet(
                    id=str(idx),
                    score=None,
                    snippet=doc,
                    metadata=None,
                )
                for idx, doc in enumerate(documents)
            ],
            fallback=bool(documents and "[NOTE:" in documents[0]),
        )
        self._persist_turn(request.conversation_id, request.question, answer)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return RAGResult(payload=payload, latency_ms=latency_ms, trace_id=trace_id)

