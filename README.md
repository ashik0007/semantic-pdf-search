# semantic-pdf-search

**Local, offline semantic search over any collection of PDFs.**

Built with [LlamaIndex](https://docs.llamaindex.ai), [Qdrant](https://qdrant.tech), and [Ollama](https://ollama.com). No cloud APIs, no data leaves your machine.

---

## What it does

- **Indexes** PDFs into a persistent local vector database using an embedding model
- **Retrieves** the most semantically relevant passages for any natural-language query
- **Synthesizes** a grounded answer using a locally running LLM — optional
- Runs entirely **offline** after a one-time model download

### Designed for

- Large technical documents: specs, manuals, standards, research papers
- Private documents you cannot send to a cloud API
- Anyone who wants a fully self-hosted document search system

### Limitations

- Default embedding models are English-only (`bge-*-en-*`). See [ADVANCED.md](ADVANCED.md) for alternatives.
- Scanned / image-only PDFs require `pymupdf` (see Troubleshooting).
- Tested with collections up to ~500 documents. For larger sets, use Qdrant server mode — see [ADVANCED.md](ADVANCED.md).

---

## Project structure

```
semantic-pdf-search/
├── pdfs/                   ← Put your PDF files here
├── index_docs.py           ← Step 1: embed and store your PDFs
├── query_docs.py           ← Step 2: query the index
├── config.py               ← All tunable parameters in one place
├── requirements.txt
├── ADVANCED.md             ← Customization, CLI flags, architecture details
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
| Python | **3.10 or 3.11** | Runtime (3.13 not yet supported) |
| Git | any | Clone this repo |
| Ollama | latest | Run local LLM |
| ~8 GB disk | — | PyTorch (~2GB) + Ollama model (~4GB) + index |

> **Apple Silicon (M1/M2/M3)** is fully supported — PyTorch uses the MPS backend automatically.

---

## Setup (step by step)

### 1 — Clone the repository

```bash
git clone https://github.com/ashik0007/semantic-pdf-search.git
cd semantic-pdf-search
```

### 2 — Install Python 3.11 and create a virtual environment

Confirm Python 3.11 is available:

```bash
python3.11 --version
```

If you see `command not found`:

```bash
brew install python@3.11                          # macOS
sudo apt install python3.11 python3.11-venv       # Ubuntu / Debian
```

Create and activate the virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** First install downloads ~2GB (includes PyTorch). This takes 3–8 minutes
> depending on your connection. **While it runs, open a new terminal and continue
> with steps 4 and 5 in parallel.**
>
> CPU-only install (smaller download, no GPU acceleration):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```
>
> You may see a HuggingFace Hub token warning — this is harmless, ignore it.

### 4 — Install and start Ollama

Download Ollama from [https://ollama.com](https://ollama.com).

**macOS (Ollama.app):** Starts automatically at login. Verify it is running:
```bash
curl http://localhost:11434
# Expected: Ollama is running
```

**Linux or macOS (terminal only):** Start manually in a separate terminal:
```bash
ollama serve
```

Once Ollama is running, pull the default model (~4GB):
```bash
ollama pull mistral
```

### 5 — Add your PDFs

Copy your PDFs into the `pdfs/` directory:

```bash
cp /path/to/your/document.pdf pdfs/
```

No PDF handy? Download a small public document to test with:

```bash
curl -L -o pdfs/test.pdf https://arxiv.org/pdf/1706.03762.pdf
file pdfs/test.pdf   # must say "PDF document", not "HTML"
```

### 6 — Index your PDFs

```bash
python index_docs.py
```

This downloads the embedding model on first run (~130MB), parses your PDFs,
and stores everything in `qdrant_storage/`.

**Expected output** (numbers depend on your document and hardware):

```
[embed] Loading embedding model: BAAI/bge-small-en-v1.5
[embed] Model ready in Xs.
[qdrant] Using local storage: /your/path/qdrant_storage
[load] Found 1 PDF(s) in '/your/path/pdfs':
       • test.pdf
[load] Loaded N document page(s).
[index] Chunking with size=512, overlap=100. Embedding and storing...
100%|████████████████████████| N/N [...]
[index] Done in Xs.
[index] Collection 'pdf_docs' is ready for queries.
        Storage: qdrant_storage
        Next step: python query_docs.py "your question here"
```
Indexing time: roughly **1–5 minutes per 100 pages** on an M3 Mac with the default model.
You may also see model loading diagnostics (e.g., `BertModel LOAD REPORT`) and a `QdrantClient` cleanup warning at exit — both are harmless.
**To add more PDFs later:** copy them into `pdfs/` and re-run `python index_docs.py`.
New documents are appended without disturbing the existing index.

> ⚠️ Use `--reset` only when rebuilding from scratch (e.g., after changing the
> embedding model in `config.py`). It permanently deletes all indexed data.

### 7 — Query your documents

```bash
python query_docs.py "Your question here"
```

**Useful options:**

```bash
python query_docs.py "question" --top-k 8       # retrieve more chunks
python query_docs.py "question" --chunks-only   # skip LLM, show raw passages only
python query_docs.py --interactive              # REPL for multiple queries
python query_docs.py "question" --out out.md    # save results to file
```

**Example output:**

```
════════════════════════════════════════════════════════════
QUERY: What is the main contribution of this paper?
════════════════════════════════════════════════════════════

── SYNTHESIZED ANSWER ──────────────────────────────────────
The paper introduces the Transformer, a model architecture based
entirely on attention mechanisms ...

── SOURCE CHUNKS (top 5) ────────────────────────────────────

[1] test.pdf  |  page 2  |  score 0.8912
    We propose a new simple network architecture, the Transformer ...
```

**What to try next:**
- Run `--interactive` mode to explore your document conversationally
- If results feel off, increase `SIMILARITY_TOP_K` in `config.py`
- Use `--chunks-only` to verify exactly what passages the system found
- See [ADVANCED.md](ADVANCED.md) to switch models, tune parameters, and scale up

---

## Troubleshooting

### `command not found: python3.11`
Install Python 3.11: `brew install python@3.11` (macOS) or `sudo apt install python3.11` (Ubuntu).

### `ModuleNotFoundError: No module named 'llama_index'`
Activate the virtual environment: `source venv/bin/activate`

### `pip install` fails with `ResolutionImpossible`

**On Python 3.13:** Not supported. Create a fresh venv with Python 3.11:
```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**On Python 3.11/3.10:** Ensure you are on the latest `requirements.txt` from the
`main` branch. If the error persists, [open an issue](https://github.com/ashik0007/semantic-pdf-search/issues)
with the full output.

### `ollama: connection refused`
Ollama is not running. On Linux: `ollama serve`. On macOS: launch the Ollama app.

### `Collection 'pdf_docs' not found`
Run `python index_docs.py` first.

### `No PDF files found in './pdfs'`
Place at least one `.pdf` file in `pdfs/`. The `PUT_YOUR_PDFS_HERE.md` placeholder
is intentionally ignored by the indexer.

### PDF text is garbled or empty
Some PDFs are scanned images with no embedded text. Install `pymupdf`:
```bash
pip install pymupdf
```
LlamaIndex will use it automatically.

### Query results are irrelevant
- Rephrase the query to use the same terminology as the document
- Increase `SIMILARITY_TOP_K` in `config.py`
- Try a larger embedding model — see [ADVANCED.md](ADVANCED.md)

### Query is slow (>60s)
- First query is always slower (LLM loads into memory). Subsequent queries are faster.
- Switch to a faster model: `OLLAMA_LLM = "phi3"` in `config.py`, then `ollama pull phi3`
- Use `--chunks-only` to skip LLM synthesis entirely

---

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- [LlamaIndex](https://github.com/run-llama/llama_index) — orchestration framework
- [Qdrant](https://github.com/qdrant/qdrant) — vector database
- [Ollama](https://github.com/ollama/ollama) — local LLM serving
- [BAAI/bge models](https://huggingface.co/BAAI) — embedding models
