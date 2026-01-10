"""
gemini_client.py
Centralized Gemini LLM and embedding configuration.
Both baseline and ontology-enhanced pipelines import from here
to guarantee identical model settings for fair comparison.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Model Configuration (single source of truth) ──
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
TEMPERATURE = 0.0  # deterministic for reproducibility
TOP_K_RETRIEVAL = 5


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance with fixed settings."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a GoogleGenerativeAIEmbeddings instance."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
