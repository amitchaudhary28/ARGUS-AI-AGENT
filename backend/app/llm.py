"""Central Gemini provider configuration for ARGUS."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env", override=False)

GEMINI_API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()


def require_gemini_api_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key is missing. Set GOOGLE_API_KEY (recommended) "
            "or GEMINI_API_KEY in backend/.env."
        )
    return GEMINI_API_KEY


@lru_cache(maxsize=4)
def get_chat_model(max_output_tokens: int = 1200) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=require_gemini_api_key(),
        temperature=0,
        max_output_tokens=max_output_tokens,
        timeout=30,
        max_retries=0,
    )
