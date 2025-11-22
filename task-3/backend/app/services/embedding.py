from __future__ import annotations

import hashlib
import threading
from typing import Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Lazy SentenceTransformer wrapper with simple text caching."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, Tuple[str, list]] = {}
        self._lock = threading.Lock()

    def _load_model(self) -> SentenceTransformer:
        """Instantiate the transformer the first time we need it (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # NOTE: We intentionally avoid 'trust_remote_code' here because the
                    # installed sentence-transformers version does not accept that
                    # keyword argument. The configured model_name (e.g. Qwen/Qwen3-Embedding-0.6B)
                    # is still respected.
                    self._model = SentenceTransformer(
                        self._model_name,
                        device=self._device,
                    )
        return self._model

    @staticmethod
    def _hash_text(text: str) -> str:
        """Content hash used as the cache key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list:
        """
        Generate (or retrieve) a normalized embedding for the provided review snippet.
        """
        text = text.strip()
        if not text:
            raise ValueError("Review text cannot be empty when generating embeddings.")

        text_hash = self._hash_text(text)

        if text_hash in self._cache:
            return self._cache[text_hash][1]

        model = self._load_model()
        vector = model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()

        with self._lock:
            self._cache[text_hash] = (text, vector)
        return vector

