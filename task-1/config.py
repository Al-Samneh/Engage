import os
import functools
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

DB_PATH = "./db"
COLLECTION_NAME = "restaurants"

# the options for the LLM were Gemini-2.5-flash-lite and Gemini-2.5-flash,
# I chose the lite due to no major difference in performance, but a major one in speed.
LLM_MODEL_NAME = "models/gemini-2.5-flash-lite"  # latency-friendly, still accurate for extraction
EMBED_MODEL_NAME = "models/gemini-embedding-001" 

# implementing a cache to avoid hitting the auth endpoint on every node invocation and keeps latency predictable.
@functools.lru_cache(maxsize=None)
def get_llm():
    """
    Reuse a single chat model instance. This avoids hitting the auth endpoint on
    every node invocation and keeps latency predictable.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
        
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=0,
        google_api_key=api_key
    )

@functools.lru_cache(maxsize=None)
def get_embeddings():
    """
    Factory for the embedding client. Caching mirrors the LLM behavior so ingest
    and retrieval do not rebuild clients mid-run.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
        
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL_NAME,
        google_api_key=api_key
    )

@functools.lru_cache(maxsize=None)
def get_chroma_client():
    """
    PersistentClient keeps the sqlite-backed collection on disk so we can round-trip
    between ingestion and LangGraph sessions without re-embedding.
    """
    return chromadb.PersistentClient(path=DB_PATH)