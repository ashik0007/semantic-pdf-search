"""
config.py — Central configuration for semantic-pdf-search.

Edit this file to switch models, tune chunking, or change storage paths.
No other file needs to change for typical customizations.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

# Directory containing your PDF files (relative to project root)
PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")

# Where Qdrant persists its vector index on disk
QDRANT_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "qdrant_storage")

# Qdrant collection name (change if you want multiple independent indexes)
QDRANT_COLLECTION_NAME = "pdf_docs"

# ─────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────
# These run locally via HuggingFace — no API key needed.
#
# Options (tradeoff: speed vs. accuracy):
#   "BAAI/bge-small-en-v1.5"   → 384-dim, ~130MB, fastest         ← default
#   "BAAI/bge-base-en-v1.5"    → 768-dim, ~440MB, better recall
#   "BAAI/bge-large-en-v1.5"   → 1024-dim, ~1.3GB, best recall (slow on CPU)
#   "sentence-transformers/all-MiniLM-L6-v2" → 384-dim, lightweight alternative
#
# NOTE: If you change this after indexing, you MUST re-index. Vectors are
# not interchangeable across embedding models.

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────
# LLM (for answer synthesis)
# ─────────────────────────────────────────────
# Served locally via Ollama (https://ollama.com).
# Pull a model first:  ollama pull <model_name>
#
# Options (tradeoff: speed vs. reasoning quality):
#   "mistral"       → 7B, ~4GB, good balance             ← default
#   "llama3"        → 8B, ~4.7GB, stronger reasoning
#   "phi3"          → 3.8B, ~2.3GB, fast, surprisingly capable
#   "orca-mini"     → 3B, ~2GB, fastest, less accurate
#   "llama3.1"      → 8B, improved instruction following
#   "gemma2"        → 9B, ~5.5GB, strong for technical text
#
# Set OLLAMA_LLM to None to skip answer synthesis and return raw chunks only.

OLLAMA_LLM = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120.0  # seconds; increase for large contexts or slow hardware

# ─────────────────────────────────────────────
# Indexing parameters
# ─────────────────────────────────────────────

# Set to True to scan subdirectories inside pdfs/ recursively
RECURSIVE_SCAN = False
CHUNK_SIZE = 512        # tokens per chunk; larger = more context, less precision
CHUNK_OVERLAP = 100     # token overlap between chunks; preserves continuity

# ─────────────────────────────────────────────
# Query parameters
# ─────────────────────────────────────────────

SIMILARITY_TOP_K = 5    # number of chunks retrieved per query
RESPONSE_MODE = "compact"  # "compact" | "tree_summarize" | "no_text" (chunks only)

# ─────────────────────────────────────────────
# Qdrant mode
# ─────────────────────────────────────────────
# "local"  → embedded Qdrant, no server needed (default, good for <1M docs)
# "server" → connect to a running Qdrant instance (scalable, production-ready)
#            requires: docker run -p 6333:6333 qdrant/qdrant
#            set QDRANT_SERVER_URL accordingly

QDRANT_MODE = "local"
QDRANT_SERVER_URL = "http://localhost:6333"  # only used when QDRANT_MODE == "server"
