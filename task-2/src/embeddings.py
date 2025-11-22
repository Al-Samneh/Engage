"""
Sentence-transformer embeddings for review text with a simple content hash cache.
"""
import hashlib
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR, DEVICE

MODEL_NAME = os.environ.get("RATING_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
CACHE_FILE = os.path.join(CACHE_DIR, "review_embeddings.npz")


def _hash_reviews(texts: pd.Series) -> str:
    """
    Build a deterministic fingerprint of all review texts. If the corpus changes,
    the hash changes, signaling that cached embeddings are stale.
    """
    hasher = hashlib.md5()
    for text in texts.astype(str):
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def _load_cached_embeddings(expected_hash: str):
    """
    Return embeddings from disk when the hash matches the current dataset.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    cache = np.load(CACHE_FILE, allow_pickle=True)
    cached_hash = cache["text_hash"].item()
    if cached_hash != expected_hash:
        return None
    return cache["embeddings"]


def _save_embeddings(embeddings: np.ndarray, text_hash: str):
    """
    Persist embedding matrix + hash for later runs.
    """
    np.savez(CACHE_FILE, embeddings=embeddings, text_hash=text_hash)


def add_embeddings(df_merged: pd.DataFrame, force_recompute: bool = False) -> Tuple[pd.DataFrame, list]:
    """
    Append dense review embeddings to the merged feature table and return the new columns.
    """
    texts = df_merged["review_text"].astype(str)
    text_hash = _hash_reviews(texts)

    if not force_recompute:
        cached = _load_cached_embeddings(text_hash)
        if cached is not None:
            print(f"[EMB] Loaded cached embeddings from {CACHE_FILE}")
            emb = cached
        else:
            emb = _compute_and_cache_embeddings(texts.tolist(), text_hash)
    else:
        emb = _compute_and_cache_embeddings(texts.tolist(), text_hash)

    dim = emb.shape[1]
    cols = [f"embed_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(emb, columns=cols, index=df_merged.index)
    df_final = pd.concat([df_merged, emb_df], axis=1)
    return df_final, cols


def _compute_and_cache_embeddings(text_list, text_hash):
    """
    Run the sentence-transformer model, normalize embeddings, then cache them for next time.
    """
    print(f"[EMB] Loading model {MODEL_NAME} on {DEVICE}")
    embedder = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    print(f"[EMB] Encoding {len(text_list)} reviews...")
    emb = embedder.encode(
        text_list,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    _save_embeddings(emb, text_hash)
    return emb

