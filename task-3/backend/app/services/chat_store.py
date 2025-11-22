from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional
from uuid import UUID


class ChatStore(ABC):
    @abstractmethod
    def fetch(self, conversation_id: UUID, limit: int = 6) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def append(
        self, conversation_id: UUID, user_turn: str, ai_turn: str, max_history: int = 6
    ) -> None:
        raise NotImplementedError


class InMemoryChatStore(ChatStore):
    """Simple in-process chat buffer for development."""

    def __init__(self) -> None:
        self._store: Dict[UUID, Deque[str]] = defaultdict(lambda: deque(maxlen=6))
        self._lock = threading.Lock()

    def fetch(self, conversation_id: UUID, limit: int = 6) -> List[str]:
        with self._lock:
            history = self._store.get(conversation_id)
            if not history:
                return []
            return list(history)[-limit:]

    def append(
        self, conversation_id: UUID, user_turn: str, ai_turn: str, max_history: int = 6
    ) -> None:
        with self._lock:
            history = self._store[conversation_id]
            history.extend([user_turn, ai_turn])
            history = deque(list(history)[-max_history:], maxlen=max_history)
            self._store[conversation_id] = history

