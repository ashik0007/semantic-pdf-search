"""
index_docs.py — Index all PDFs in the configured pdf directory into Qdrant.

Usage:
    python index_docs.py                 # index everything in pdfs/
    python index_docs.py --reset         # wipe the existing index first, then re-index
    python index_docs.py --pdf path.pdf  # index a single file (adds to existing index)

Run this once (or after adding new PDFs). Indexing is idempotent when using
the same collection name — existing documents are overwritten, not duplicated.
"""

import argparse
import os
import sys
import time

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def build_qdrant_client() -> QdrantClient:
    """Return a QdrantClient based on config.QDRANT_MODE."""
    if config.QDRANT_MODE == "server":
        print(f"[qdrant] Connecting to server: {config.QDRANT_SERVER_URL}")
        return QdrantClient(url=config.QDRANT_SERVER_URL)
    else:
        os.makedirs(config.QDRANT_STORAGE_PATH, exist_ok=True)
        print(f"[qdrant] Using local storage: {config.QDRANT_STORAGE_PATH}")
        return QdrantClient(path=config.QDRANT_STORAGE_PATH)


def get_embedding_dim(embed_model) -> int:
    """Probe the embedding model for its output dimension."""
    sample = embed_model.get_text_embedding("probe")
    return len(sample)


def reset_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Delete and recreate the Qdrant collection."""
    if client.collection_exists(name):
        print(f"[qdrant] Deleting existing collection '{name}'...")
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"[qdrant] Created fresh collection '{name}' (dim={dim}).")


def load_documents(source: str):
    """
    Load documents from a directory or single file.

    Returns a list of LlamaIndex Document objects.
    """
    if os.path.isfile(source):
        if not source.lower().endswith(".pdf"):
            print(f"[warn] '{source}' does not appear to be a PDF. Proceeding anyway.")
        reader = SimpleDirectoryReader(input_files=[source])
    elif os.path.isdir(source):
        pdf_files = [
            f for f in os.listdir(source) if f.lower().endswith(".pdf")
        ]
        if not pdf_files:
            print(f"[error] No PDF files found in '{source}'.")
            sys.exit(1)
        print(f"[load] Found {len(pdf_files)} PDF(s) in '{source}':")
        for f in sorted(pdf_files):
            print(f"       • {f}")
        reader = SimpleDirectoryReader(
            input_dir=source,
            required_exts=[".pdf"],
            recursive=False,
        )
    else:
        print(f"[error] Source '{source}' is neither a file nor a directory.")
        sys.exit(1)

    docs = reader.load_data()
    print(f"[load] Loaded {len(docs)} document page(s).")
    return docs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index PDF files into a local Qdrant vector store."
    )
    parser.add_argument(
        "--pdf",
        metavar="PATH",
        default=None,
        help="Path to a single PDF file. Defaults to config.PDF_DIR.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing Qdrant collection before indexing.",
    )
    args = parser.parse_args()

    source = args.pdf if args.pdf else config.PDF_DIR

    # ── Embedding model ──────────────────────────────────────────────────────
    print(f"[embed] Loading embedding model: {config.EMBED_MODEL_NAME}")
    t0 = time.time()
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    print(f"[embed] Model ready in {time.time() - t0:.1f}s.")

    # Wire the embedding model globally so LlamaIndex picks it up
    Settings.embed_model = embed_model
    Settings.llm = None  # indexing does not need an LLM

    # ── Qdrant client ────────────────────────────────────────────────────────
    client = build_qdrant_client()
    embed_dim = get_embedding_dim(embed_model)

    if args.reset:
        reset_collection(client, config.QDRANT_COLLECTION_NAME, embed_dim)
    elif not client.collection_exists(config.QDRANT_COLLECTION_NAME):
        print(f"[qdrant] Collection '{config.QDRANT_COLLECTION_NAME}' not found. Creating.")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Load documents ───────────────────────────────────────────────────────
    documents = load_documents(source)

    # ── Chunk and index ──────────────────────────────────────────────────────
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    print(
        f"[index] Chunking with size={config.CHUNK_SIZE}, "
        f"overlap={config.CHUNK_OVERLAP}. Embedding and storing..."
    )
    t1 = time.time()

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    elapsed = time.time() - t1
    print(f"\n[index] Done in {elapsed:.1f}s.")
    print(
        f"[index] Collection '{config.QDRANT_COLLECTION_NAME}' is ready for queries."
    )
    print(f"        Storage: {config.QDRANT_STORAGE_PATH}")
    print(f"        Next step: python query_docs.py \"your question here\"")


if __name__ == "__main__":
    main()
