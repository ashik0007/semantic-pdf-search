# Advanced Configuration & Reference

This document covers customization, tuning, CLI flags, and architecture details
for `semantic-pdf-search`. If you are setting up the project for the first time,
start with the [README](README.md).

---

## Customization

All parameters live in **`config.py`** — you do not need to edit the other scripts.

### Switch the embedding model

```python
# config.py
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"   # larger, more accurate
```

| Model | Dimension | Size | Speed (M3) | Notes |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~130MB | fastest | **default** |
| `BAAI/bge-base-en-v1.5` | 768 | ~440MB | moderate | better recall |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3GB | slow on CPU | best recall |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90MB | fastest | lightweight alternative |

> ⚠️ Changing the embedding model requires rebuilding the entire index:
> ```bash
> python index_docs.py --reset
> ```
> Vectors from different embedding models are not interchangeable.

### Switch the LLM

```python
# config.py
OLLAMA_LLM = "llama3"   # then: ollama pull llama3
```

| Model | Size | Speed (M3) | Notes |
|---|---|---|---|
| `mistral` | ~4GB | ~8s/query | **default**, good balance |
| `llama3` | ~4.7GB | ~10s/query | stronger reasoning |
| `phi3` | ~2.3GB | ~4s/query | fast, surprisingly capable |
| `orca-mini` | ~2GB | ~3s/query | fastest, less accurate |
| `gemma2` | ~5.5GB | ~12s/query | strong for technical text |
| `llama3.1` | ~4.7GB | ~10s/query | improved instruction following |

To disable LLM synthesis and return raw chunks only:
```python
OLLAMA_LLM = None
```

### Tune chunking

```python
# config.py
CHUNK_SIZE = 512     # tokens per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks
```

Larger chunks → more context per retrieval, less precise matching.
Smaller chunks → more precise matching, less surrounding context.
Typical range: `CHUNK_SIZE` 256–1024.

### Tune retrieval

```python
# config.py
SIMILARITY_TOP_K = 5      # chunks retrieved per query
RESPONSE_MODE = "compact" # "compact" | "tree_summarize" | "no_text"
```

- `compact` — concatenates all retrieved chunks and passes them to the LLM in one call (default, fast)
- `tree_summarize` — builds a hierarchical summary across chunks; better for long or multi-section answers, but slower
- `no_text` — skips the LLM entirely and returns raw chunks only; same as `--chunks-only` on the CLI

### Scan PDFs in subdirectories

```python
# config.py
RECURSIVE_SCAN = True   # scan all subdirectories inside pdfs/
```

### Use Qdrant as a server (large collections or multi-user)

The default embedded Qdrant works well up to ~50,000 chunks. For larger collections
or shared access, run Qdrant as a standalone server via Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then in `config.py`:
```python
QDRANT_MODE = "server"
QDRANT_SERVER_URL = "http://localhost:6333"
```

---

## Understanding similarity scores

Query results show a score like `score 0.8912`. This is cosine similarity between
the query embedding and the chunk embedding, ranging from 0 to 1:

| Score range | Interpretation |
|---|---|
| 0.85 – 1.0 | Strong match — chunk is highly relevant |
| 0.70 – 0.85 | Good match — likely relevant |
| 0.50 – 0.70 | Weak match — may be tangentially related |
| < 0.50 | Poor match — probably not relevant |

If your top results are all below 0.60, try:
- Rephrasing the query to match the document's terminology
- Increasing `SIMILARITY_TOP_K` to cast a wider net
- Switching to a larger embedding model

---

## Handling LLM hallucinations

The LLM synthesizes answers from retrieved chunks — it can still produce confident
but incorrect statements, especially if the relevant content is not in the top-K
chunks. To verify any answer:

1. Run with `--chunks-only` to see the raw source passages
2. Check the source filename and page number shown for each chunk
3. Increase `--top-k` if the answer seems to be missing context

The `--chunks-only` flag bypasses the LLM entirely and returns only the retrieved
text — use it when you want to verify exactly what the system found.

---

## CLI reference

### `index_docs.py`

```
python index_docs.py                    # index all PDFs in pdfs/
python index_docs.py --reset            # ⚠️ wipe entire index, rebuild from scratch
python index_docs.py --pdf myfile.pdf   # index a single file (adds to existing index)
```

> ⚠️ `--reset` permanently deletes all indexed data. Use it only when changing the
> embedding model or starting fresh. To add new PDFs, just run without `--reset`.

### `query_docs.py`

```
python query_docs.py "question"                      # single query with LLM synthesis
python query_docs.py "question" --top-k 10           # retrieve 10 chunks
python query_docs.py "question" --chunks-only        # no LLM, raw chunks only
python query_docs.py --interactive                   # REPL mode for multiple queries
python query_docs.py "question" --out results.md     # append output to file
```

---

## How it works

```
┌─────────────┐    chunk + embed     ┌──────────────────┐
│  PDF Files  │ ─────────────────→  │ Qdrant (local DB) │
└─────────────┘  BAAI/bge-small     └──────────────────┘
                                              │
                                     similarity search
                                              │
   ┌──────────┐    natural language   ┌───────▼────────┐
   │  Query   │ ─────────────────→  │ Top-K Chunks   │
   └──────────┘   same embed model   └───────┬────────┘
                                              │
                                    LLM synthesis
                                    (Ollama / mistral)
                                              │
                                     ┌────────▼────────┐
                                     │  Answer + Sources│
                                     └─────────────────┘
```

1. **Indexing**: PDFs are parsed into pages → split into overlapping token chunks →
   each chunk is converted to a dense vector by the embedding model → stored in Qdrant.

2. **Querying**: The query is embedded using the same model → Qdrant performs a
   cosine similarity search and returns the top-K most similar chunks → the chunks
   are passed to the LLM with a prompt asking it to synthesize a grounded answer.

### Why chunking matters

PDFs are too long to embed as a whole. Chunking splits them into segments small
enough to embed meaningfully. The `CHUNK_OVERLAP` parameter ensures that sentences
split across chunk boundaries are still captured by at least one chunk.

### Why local embeddings

Embedding models convert text into vectors that capture semantic meaning — "car"
and "automobile" will have similar vectors even though they share no characters.
This allows the system to find relevant passages even when the exact query words
don't appear in the document.
