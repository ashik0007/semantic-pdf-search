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
├── pdfs/                   ← Put your PDF files here (PUT_YOUR_PDFS_HERE.md is a placeholder, safe to ignore)
├── index_docs.py           ← Step 1: embed and store your PDFs
├── query_docs.py           ← Step 2: query the index
├── config.py               ← All tunable parameters in one place
├── requirements.txt
├── .gitignore
├── LICENSE
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
| Python | **3.10 or 3.11** | Runtime (3.13 not supported) |
| Git | any | Clone this repo |
| Ollama | latest | Run local LLM |
| ~5 GB disk | — | Models + index |

> **Python 3.13 is not supported.** The LlamaIndex 0.11.x packages this project depends on
> require Python ≤ 3.11. See Step 2 for how to install a compatible version.
>
> **Apple Silicon (M1/M2/M3)** is fully supported — PyTorch uses the MPS backend automatically.

---

## Setup (step by step)

### 1 — Clone the repository
```bash
git clone https://github.com/ashik0007/semantic-pdf-search.git
cd semantic-pdf-search
```

### 2 — Install Python 3.11 and create a virtual environment

First, confirm Python 3.11 is available:
```bash
python3.11 --version
```

If you see `command not found`, install it first:
```bash
# macOS (Homebrew)
brew install python@3.11

# Ubuntu / Debian
sudo apt install python3.11 python3.11-venv
```

Then create and activate the virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (Command Prompt)
```

Confirm the version inside the venv:
```bash
python --version    # must print Python 3.11.x
```

### 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch size:** `torch` is pulled in by `sentence-transformers` (~2 GB on first install).
> If you only need CPU inference and want a smaller download, pre-install a CPU-only wheel first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4 — Install and start Ollama

Download Ollama from [https://ollama.com](https://ollama.com) and install it for your platform.

Pull a model. `mistral` is the default in `config.py`:
```bash
ollama pull mistral
```

**Starting Ollama — pick your platform:**

**macOS (Ollama.app):** Ollama starts automatically at login. Skip `ollama serve`. Just verify:
```bash
curl http://localhost:11434
# Expected: Ollama is running
```

**Linux or macOS (terminal only):** Start the server manually in a separate terminal window:
```bash
ollama serve
```
Leave that terminal open and proceed in a new one.

### 5 — Add your PDFs

Copy your PDF files into the `pdfs/` directory:
```bash
cp /path/to/your/document.pdf pdfs/
```

Don't have a PDF ready? Download a small reliable public document to test with:
```bash
# "Attention Is All You Need" — the transformer paper (~2MB, stable URL)
curl -L -o pdfs/test.pdf https://arxiv.org/pdf/1706.03762.pdf

# Verify it downloaded correctly (must say "PDF document", not "HTML")
file pdfs/test.pdf
# Expected: pdfs/test.pdf: PDF document, version 1.x
```

> If `file` prints `HTML document`, the download failed. Delete the file and try again
> or use your own PDF.

### 6 — Index your PDFs
```bash
python index_docs.py
```

This will:
1. Download the embedding model on first run (~130MB for `bge-small-en-v1.5`)
2. Parse all PDFs and split them into overlapping 512-token chunks
3. Embed each chunk and store it in `qdrant_storage/`

**Expected output** (exact numbers depend on your document and hardware):
```
[embed] Loading embedding model: BAAI/bge-small-en-v1.5
[embed] Model ready in Xs.      ← first run downloads ~130MB; later runs are instant
[qdrant] Using local storage: ./qdrant_storage
[load] Found 1 PDF(s) in './pdfs':
       • test.pdf
[load] Loaded N document page(s).
[index] Chunking with size=512, overlap=100. Embedding and storing...
100%|████████████████████████| N/N [...]
[index] Done in Xs.
[index] Collection 'pdf_docs' is ready for queries.
        Next step: python query_docs.py "your question here"
```

> Indexing time: roughly **1–5 minutes per 100 pages** on an M3 Mac with the default model.

**Adding more PDFs later:**
Simply copy them into `pdfs/` and re-run `python index_docs.py`. New documents are
appended to the existing index without disturbing what is already there.

> ⚠️ **`--reset` deletes all indexed data.** Use it only when rebuilding from scratch
> (e.g., after changing the embedding model). Never use `--reset` just to add new PDFs.

### 7 — Query your documents
```bash
python query_docs.py "Your question here"
```

**Examples:**
```bash
python query_docs.py "What is the main contribution of this paper?"
python query_docs.py "Explain the attention mechanism" --top-k 8
python query_docs.py "transformer architecture" --chunks-only   # skip LLM, raw chunks only
python query_docs.py --interactive                               # REPL mode
python query_docs.py "some query" --out results.md              # save to file
```

**Example output:**
```
════════════════════════════════════════════════════════════
QUERY: What is the main contribution of this paper?
════════════════════════════════════════════════════════════

── SYNTHESIZED ANSWER ──────────────────────────────────────
The paper introduces the Transformer, a model architecture based
entirely on attention mechanisms, dispensing with recurrence and
convolutions entirely ...

── SOURCE CHUNKS (top 5) ────────────────────────────────────

[1] test.pdf  |  page 2  |  score 0.8912
    We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms ...
```

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

> ⚠️ Changing the embedding model requires rebuilding the index:
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

- `compact` — concatenates chunks, one LLM call (default, fast)
- `tree_summarize` — hierarchical summarization (better for long answers, slower)
- `no_text` — skips LLM, same as `--chunks-only`

### Scan PDFs in subdirectories

In `index_docs.py`, add `recursive=True`:
```python
reader = SimpleDirectoryReader(
    input_dir=source,
    required_exts=[".pdf"],
    recursive=True,       # ← add this
)
```

### Use Qdrant as a server (large collections or multi-user)
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then in `config.py`:
```python
QDRANT_MODE = "server"
QDRANT_SERVER_URL = "http://localhost:6333"
```

Recommended for collections above ~50,000 chunks.

---

## CLI reference

### `index_docs.py`
```
python index_docs.py                    # index all PDFs in pdfs/
python index_docs.py --reset            # ⚠️ wipe entire index, then re-index from scratch
python index_docs.py --pdf myfile.pdf   # index a single file (adds to existing index)
```

### `query_docs.py`
```
python query_docs.py "question"                      # single query with LLM synthesis
python query_docs.py "question" --top-k 10           # retrieve 10 chunks
python query_docs.py "question" --chunks-only        # no LLM, raw chunks only
python query_docs.py --interactive                   # REPL mode
python query_docs.py "question" --out results.md     # append output to file
```

---

## Troubleshooting

### `ollama: connection refused`
Ollama is not running. On Linux or macOS (terminal only): `ollama serve`.
On macOS, launch the Ollama app from Applications.

### `ModuleNotFoundError: No module named 'llama_index'`
The virtual environment is not activated:
```bash
source venv/bin/activate
```

### `pip install` fails with `ResolutionImpossible`

**If you are on Python 3.13:** Python 3.13 is not supported. Create a new venv with 3.11:
```bash
brew install python@3.11        # macOS
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
**If you are on Python 3.11 or 3.10 and still see this error:** The package
versions in `requirements.txt` may have drifted out of sync with PyPI. This is
a genuine version conflict — `pip cache purge` will not fix it. Please ensure
you are using the latest `requirements.txt` from the `main` branch, then
[open a GitHub issue](https://github.com/ashik0007/semantic-pdf-search/issues)
with the full error output if the problem persists.
```

**Patch B** — Find the Step 6 expected output block and replace:
```
[qdrant] Using local storage: ./qdrant_storage
```
with:
```
[qdrant] Using local storage: qdrant_storage
```

And add the Storage line to match actual code output:
[index] Collection 'pdf_docs' is ready for queries.
        Storage: qdrant_storage
        Next step: python query_docs.py "your question here"


### `Collection 'pdf_docs' not found`
The index has not been built yet:
```bash
python index_docs.py
```

### `No PDF files found in './pdfs'`
Place at least one `.pdf` file in the `pdfs/` directory. The `PUT_YOUR_PDFS_HERE.md`
placeholder is not a PDF and is intentionally ignored by the indexer.

### Test PDF downloaded as HTML (not a PDF)
The download URL may have changed. Verify with:
```bash
file pdfs/test.pdf
```
If it prints `HTML document`, delete the file and try a different source, or use your own PDF.

### Query is slow (>60s)
- First query is always slower (LLM loads into memory). Subsequent queries are faster.
- Switch to a smaller model: set `OLLAMA_LLM = "phi3"` in `config.py`, then `ollama pull phi3`.
- Use `--chunks-only` to bypass LLM synthesis entirely.

### Irrelevant results
- Increase `SIMILARITY_TOP_K` (e.g., 10) in `config.py`.
- Try a larger embedding model (`BAAI/bge-base-en-v1.5`) — remember to `--reset` and re-index.
- Reduce `CHUNK_SIZE` to 256 for finer-grained matching.

### PDF text not extracted correctly
Some PDFs are scanned images with no embedded text. Install `pymupdf` for better extraction:
```bash
pip install pymupdf
```
LlamaIndex will prefer it over `pypdf` automatically.

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

1. **Indexing**: PDFs → chunked into overlapping segments → each chunk embedded as a dense vector → stored in Qdrant.
2. **Querying**: Query embedded with the same model → Qdrant returns top-K most similar chunks (cosine similarity) → chunks passed to LLM for grounded answer synthesis.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [LlamaIndex](https://github.com/run-llama/llama_index) — RAG orchestration framework
- [Qdrant](https://github.com/qdrant/qdrant) — vector database
- [Ollama](https://github.com/ollama/ollama) — local LLM serving
- [BAAI/bge models](https://huggingface.co/BAAI) — embedding models
