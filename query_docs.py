"""
query_docs.py — Query the indexed PDF collection.

Usage:
    python query_docs.py "What is the PDCCH blind decoding limit?"
    python query_docs.py "Explain MCS table 1" --top-k 8
    python query_docs.py "slot formats" --chunks-only      # skip LLM synthesis
    python query_docs.py --interactive                     # REPL mode
    python query_docs.py "query" --out results.md          # save output to file

Requires:
    - Ollama running locally:  ollama serve
    - The chosen LLM pulled:   ollama pull mistral
    - Index already built:     python index_docs.py
"""

import argparse
import os
import sys
import textwrap
import time

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from qdrant_client import QdrantClient

import config


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def build_qdrant_client() -> QdrantClient:
    if config.QDRANT_MODE == "server":
        return QdrantClient(url=config.QDRANT_SERVER_URL)
    else:
        if not os.path.exists(config.QDRANT_STORAGE_PATH):
            print(
                f"[error] No index found at '{config.QDRANT_STORAGE_PATH}'.\n"
                f"        Run:  python index_docs.py"
            )
            sys.exit(1)
        return QdrantClient(path=config.QDRANT_STORAGE_PATH)


def load_llm():
    """Load the Ollama LLM if configured; return None for chunks-only mode."""
    if config.OLLAMA_LLM is None:
        return None
    try:
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=config.OLLAMA_LLM,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=config.OLLAMA_TIMEOUT,
        )
    except ImportError:
        print(
            "[warn] llama-index-llms-ollama not installed. "
            "Falling back to chunks-only mode."
        )
        return None


def build_query_engine(top_k: int, chunks_only: bool, llm):
    """
    Construct a LlamaIndex query engine backed by the persisted Qdrant index.

    chunks_only=True → retriever only, no LLM synthesis.
    """
    # Embedding model (must match what was used during indexing)
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.llm = llm  # may be None

    client = build_qdrant_client()

    if not client.collection_exists(config.QDRANT_COLLECTION_NAME):
        print(
            f"[error] Collection '{config.QDRANT_COLLECTION_NAME}' not found.\n"
            f"        Run:  python index_docs.py"
        )
        sys.exit(1)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    if chunks_only or llm is None:
        # Return only the retrieved chunks, no synthesis
        response_synthesizer = get_response_synthesizer(response_mode="no_text")
    else:
        response_synthesizer = get_response_synthesizer(
            response_mode=config.RESPONSE_MODE,
            llm=llm,
        )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.0)],
    )


def format_response(query: str, response, chunks_only: bool) -> str:
    """Format the query response as readable text."""
    lines = []
    lines.append(f"\n{'═' * 60}")
    lines.append(f"QUERY: {query}")
    lines.append(f"{'═' * 60}")

    if not chunks_only and response.response:
        lines.append("\n── SYNTHESIZED ANSWER ──────────────────────────────────")
        lines.append(textwrap.fill(response.response.strip(), width=80))

    lines.append(f"\n── SOURCE CHUNKS (top {len(response.source_nodes)}) ────────────────────────────")
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata
        source = meta.get("file_name", meta.get("source", "unknown"))
        page = meta.get("page_label", meta.get("page", "?"))
        score = node.score if node.score is not None else 0.0
        lines.append(f"\n[{i}] {source}  |  page {page}  |  score {score:.4f}")
        lines.append("    " + "\n    ".join(
            textwrap.wrap(node.node.get_content().strip(), width=76)
        ))

    lines.append(f"\n{'─' * 60}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query the indexed PDF collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to answer. If omitted, use --interactive.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.SIMILARITY_TOP_K,
        metavar="K",
        help=f"Number of chunks to retrieve (default: {config.SIMILARITY_TOP_K}).",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Return raw chunks without LLM synthesis.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive query REPL.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Append results to this file (Markdown-friendly).",
    )
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(0)

    # ── Setup ────────────────────────────────────────────────────────────────
    print(f"[embed] Loading embedding model: {config.EMBED_MODEL_NAME} ...")
    llm = None if args.chunks_only else load_llm()
    engine = build_query_engine(
        top_k=args.top_k,
        chunks_only=args.chunks_only,
        llm=llm,
    )
    print("[ready] Query engine initialized.\n")

    # ── Query or REPL ────────────────────────────────────────────────────────
    def run_query(q: str):
        t0 = time.time()
        response = engine.query(q)
        elapsed = time.time() - t0
        output = format_response(q, response, args.chunks_only)
        print(output)
        print(f"[timing] {elapsed:.1f}s")

        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(output)
                f.write(f"\n*Query time: {elapsed:.1f}s*\n\n")
            print(f"[saved] Appended to {args.out}")

    if args.interactive:
        print("Interactive mode. Type 'exit' or press Ctrl-C to quit.\n")
        while True:
            try:
                q = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break
            run_query(q)
    else:
        run_query(args.query)


if __name__ == "__main__":
    main()
