# semantic-pdf-search

**Local, offline semantic search over any collection of PDFs.**

Built with [LlamaIndex](https://docs.llamaindex.ai), [Qdrant](https://qdrant.tech), and [Ollama](https://ollama.com). No cloud APIs, no data leaves your machine.

---

## What it does

- **Indexes** PDFs into a persistent vector database (Qdrant) using a local embedding model
- **Retrieves** the most semantically relevant chunks for any natural-language query
- **Synthesizes** a concise answer using a locally running LLM (via Ollama) — optional
- Runs entirely **offline** after a one-time model download (~300MB–2GB depending on choices)

### Designed for

- Large technical documents (specs, manuals, standards, research papers)
- Private documents you cannot send to a cloud API
- Anyone who wants a reproducible, fully self-hosted RAG pipeline

---

## Project structure

```
semantic-pdf-search/
├── pdfs/                   ← Put your PDF files here
│   └── PUT_YOUR_PDFS_HERE.md
├── index_docs.py           ← Step 1: embed and store your PDFs
├── query_docs.py           ← Step 2: query the index
├── config.py               ← All tunable parameters in one place
├── requirements.txt
├── .gitignore
└── README.md
```

After running `index_docs.py`, Qdrant creates:
```
qdrant_storage/             ← Auto-created, persists across runs (gitignored)
```

---

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10 or 3.11 | Runtime |
| Git | any | Clone this repo |
| Ollama | latest | Run local LLM |
| ~5 GB disk | — | Models + index |

> **Python 3.12** works but some transitive dependencies emit deprecation warnings.
> **Apple Silicon (M1/M2/M3)** is fully supported — PyTorch uses the MPS backend automatically.

---

## Setup (step by step)

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/semantic-pdf-search.git
cd semantic-pdf-search
```

### 2 — Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (Command Prompt)
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch size:** `torch` is pulled transitively by `sentence-transformers` (~2 GB).
> If you only need CPU inference and want a smaller install, you can pre-install a CPU-only
> wheel before running the above:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4 — Install and start Ollama

Download Ollama from [https://ollama.com](https://ollama.com) and install it for your platform.

Then pull a model. `mistral` is the default in `config.py`:

```bash
ollama pull mistral
```

Start the Ollama server (it runs in the background):

```bash
ollama serve
```

> On macOS, Ollama starts automatically after installation.
> Verify it is running: `curl http://localhost:11434`

### 5 — Add your PDFs

Copy or move your PDF files into the `pdfs/` directory:

```bash
cp /path/to/your/document.pdf pdfs/
```

You can add as many PDFs as you like. Subdirectories are not scanned by default
(see [Customization](#customization) to enable recursive scanning).

### 6 — Index your PDFs

```bash
python index_docs.py
```

This will:
1. Download the embedding model on first run (~130MB for the default `bge-small-en-v1.5`)
2. Parse all PDFs, chunk them into 512-token segments
3. Embed each chunk and store it in `qdrant_storage/`

**Expected output:**

```
[embed] Loading embedding model: BAAI/bge-small-en-v1.5
[embed] Model ready in 3.2s.
[qdrant] Using local storage: ./qdrant_storage
[load] Found 3 PDF(s) in './pdfs':
       • document_a.pdf
       • document_b.pdf
       • document_c.pdf
[load] Loaded 1840 document page(s).
[index] Chunking with size=512, overlap=100. Embedding and storing...
100%|████████████████████████| 5535/5535 [03:12<00:00, 28.8it/s]
[index] Done in 192.4s.
[index] Collection 'pdf_docs' is ready for queries.
```

> Indexing time: roughly **1–5 minutes per 100 pages** on an M3 Mac with the default model.
> Re-runs are fast if you add new PDFs with `--reset` (see below).

### 7 — Query your documents

```bash
python query_docs.py "Your question here"
```

**Examples:**

```bash
python query_docs.py "What is the maximum number of PDCCH candidates per slot?"
python query_docs.py "Explain the difference between MCS table 1 and table 2" --top-k 8
python query_docs.py "slot formats TDD" --chunks-only     # skip LLM, show raw chunks
python query_docs.py --interactive                         # REPL mode for multiple queries
python query_docs.py "some query" --out results.md         # save to file
```

**Example output:**

```
════════════════════════════════════════════════════════════
QUERY: What is the maximum number of PDCCH candidates per slot?
════════════════════════════════════════════════════════════

── SYNTHESIZED ANSWER ──────────────────────────────────────
The maximum number of PDCCH candidates per slot is 44 for aggregation
level 1, as specified in TS 38.213 Table 10.1-2. This limit applies
across all configured search spaces.

── SOURCE CHUNKS (top 5) ────────────────────────────────────

[1] TS_38_212_v17.pdf  |  page 47  |  score 0.8932
    The maximum number of PDCCH candidates per slot per serving cell
    is given in Table 10.1-2 ...

[2] TS_38_211_v17.pdf  |  page 23  |  score 0.8741
    ...
```

---

## Customization

All parameters live in **`config.py`** — you should not need to edit the other scripts.

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

> **Important:** If you change the embedding model, you must re-index:
> ```bash
> python index_docs.py --reset
> ```
> Vectors from different models are incompatible.

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

To disable LLM synthesis entirely (return raw chunks only):

```python
OLLAMA_LLM = None
```

### Tune chunking

```python
# config.py
CHUNK_SIZE = 512     # tokens per chunk; increase for more context per chunk
CHUNK_OVERLAP = 100  # token overlap; increase if answers are cut off at boundaries
```

Larger chunks → more context per retrieval, but less precise matching.
Smaller chunks → more precise matching, but answers may lack context.
Typical range: `CHUNK_SIZE` 256–1024.

### Tune retrieval

```python
# config.py
SIMILARITY_TOP_K = 5      # how many chunks to retrieve per query
RESPONSE_MODE = "compact" # "compact" | "tree_summarize" | "no_text"
```

- `compact` — concatenates chunks, passes to LLM once (default, fast)
- `tree_summarize` — hierarchical summarization (better for long answers, slower)
- `no_text` — same as `--chunks-only`, skips LLM

### Add PDFs from subdirectories

In `index_docs.py`, change:

```python
reader = SimpleDirectoryReader(
    input_dir=source,
    required_exts=[".pdf"],
    recursive=True,       # ← add this
)
```

### Use Qdrant as a server (for large collections or multi-user access)

Start Qdrant via Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then set in `config.py`:

```python
QDRANT_MODE = "server"
QDRANT_SERVER_URL = "http://localhost:6333"
```

The server mode is recommended for collections above ~50,000 chunks.

---

## CLI reference

### `index_docs.py`

```
python index_docs.py                    # index all PDFs in pdfs/
python index_docs.py --reset            # wipe index first, then re-index
python index_docs.py --pdf myfile.pdf   # index a single file (adds to index)
```

### `query_docs.py`

```
python query_docs.py "question"         # single query
python query_docs.py "question" --top-k 10          # retrieve 10 chunks
python query_docs.py "question" --chunks-only        # no LLM synthesis
python query_docs.py --interactive                   # REPL mode
python query_docs.py "question" --out results.md     # append output to file
```

---

## Troubleshooting

### `ollama: connection refused`
Ollama is not running. Start it:
```bash
ollama serve
```
Or on macOS, launch the Ollama app from Applications.

### `ModuleNotFoundError: No module named 'llama_index'`
The virtual environment is not activated:
```bash
source venv/bin/activate
```

### `Collection 'pdf_docs' not found`
You have not run the indexer yet, or the storage path has changed:
```bash
python index_docs.py
```

### `No PDF files found in './pdfs'`
Place at least one `.pdf` file in the `pdfs/` directory before indexing.

### Query is very slow (>60s)
- The LLM is loading for the first time. Subsequent queries are faster.
- Switch to a smaller model: `OLLAMA_LLM = "phi3"` in `config.py`.
- Use `--chunks-only` to skip LLM synthesis entirely.

### Irrelevant results
- Increase `SIMILARITY_TOP_K` to retrieve more candidates.
- Try a larger embedding model (`bge-base-en-v1.5`).
- Reduce `CHUNK_SIZE` (e.g., to 256) to improve chunk granularity.

### PDF text not extracted correctly
Some PDFs are scanned images. Install `pymupdf` for better extraction:
```bash
pip install pymupdf
```
LlamaIndex will prefer it over `pypdf` automatically when installed.

---

## How it works (architecture)

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

1. **Indexing**: PDFs are parsed → split into overlapping chunks → each chunk is
   converted to a dense vector by the embedding model → stored in Qdrant.

2. **Querying**: The query is embedded using the same model → Qdrant returns the
   top-K most similar chunks (cosine similarity) → the chunks are passed to the
   LLM with a prompt to synthesize a grounded answer.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [LlamaIndex](https://github.com/run-llama/llama_index) — RAG orchestration framework
- [Qdrant](https://github.com/qdrant/qdrant) — vector database
- [Ollama](https://github.com/ollama/ollama) — local LLM serving
- [BAAI/bge models](https://huggingface.co/BAAI) — embedding models# semantic-pdf-search
Local, offline semantic search over any collection of PDFs.
