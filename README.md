*This project has been created as part of the 42 curriculum by msbayihi.*

# RAG Against the Machine

## Description

This project implements a complete **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions about a real codebase . Instead of relying on a language model's frozen training data, the system retrieves the most relevant source snippets from the codebase at answer time, then uses a small local model (Qwen/Qwen3-0.6B) to generate a grounded answer from that retrieved context.

The pipeline follows the four classic RAG stages:

1. **Indexing** — the codebase is chunked (with two distinct strategies for Python code and Markdown documentation) and indexed with BM25 for fast lexical search.
2. **Retrieval** — given a question, the system returns the top-k most relevant source locations (file path + character range (first char to last cha)).
3. **Augmenting** — the retrieved snippets are assembled into a context window within the model's token budget.
4. **Generating** — Qwen3-0.6B produces a natural-language answer grounded in the retrieved context.

Retrieval quality is measured with **recall@k**, and the project includes five bonus extensions: semantic embeddings, hybrid retrieval, incremental indexing, caching, and a local HTTP API.

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) as the package and project manager

### Installation

```bash
make install
```

This runs `uv sync`, which installs all dependencies declared in `pyproject.toml`
(including the bonus dependencies: `sentence-transformers`, `fastapi`, `uvicorn`).

### Running the pipeline

All commands follow the pattern `uv run python3 -m src <command> [options]`.

**1. Build the index** (mandatory, run this first):

```bash
uv run python3 -m src index --max_chunk_size 2000
```

**2. Search for a single question:**

```bash
uv run python3 -m src search "How to configure the OpenAI server?" --k 5
```

**3. Generate an answer for a single question:**

```bash
uv run python3 -m src answer "How to configure the OpenAI server?" --k 5
```

**4. Run search over a whole dataset:**

```bash
uv run python3 -m src search_data
    for source in top_sources:set \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

**5. Generate answers for a whole dataset:**

```bash
uv run python3 -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

**6. Evaluate recall@k against a ground-truth dataset:**

```bash
uv run python3 -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
### Makefile commands

| Command            | Description                                |
| ------------------ | ------------------------------------------ |
| `make install`     | Install dependencies with `uv sync`        |
| `make run`         | Run the CLI                                |
| `make debug`       | Run the CLI with Python's debugger (`pdb`) |
| `make clean`       | Remove cache files                         |
| `make lint`        | Run `flake8` and `mypy`                    |


## Resources

### Documentation and references

- [vLLM Documentation](https://docs.vllm.ai/) — the indexed codebase
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — the generation model
- [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://bm25s.github.io/) — the BM25 library used for lexical retrieval
- [Sentence-Transformers Documentation](https://www.sbert.net/) — used for the semantic embeddings bonus (`all-MiniLM-L6-v2`)
- [LangChain Text Splitters Documentation](https://www.langchain.com/) — `RecursiveCharacterTextSplitter` used for chunking (.md, txt, py ...)
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — used for the local HTTP API bonus
- [Boot.dev](https://www.boot.dev/courses/learn-retrieval-augmented-generation) — used to learn the fundamentals of RAG and understand its main components
- [different btw tf-idf and bm25](https://emschwartz.me/comparing-full-text-search-algorithms-bm25-tf-idf-and-postgres/learn-retrieval-augmented-generation) — Comparing full text search algorithms: BM25, TF-IDF, and Postgres

### AI usage disclosure

- **Architecture design**: discussing and validating the overall pipeline
  structure (abstract base classes for chunkers and retrievers, service/
  orchestrator separation) before writing any code, so that bonus features
  could be added later without reworking the mandatory part.

- **Debugging**: diagnosing and fixing issues discovered during testing such as the Qwen3 "thinking mode" producing `<think>` blocks in generated answers, and the score-scale mismatch between BM25 and cosine similarity in the hybrid retriever.

- **Documentation**: drafting sections of this README, which were then reviewed and adjusted.

All code was tested manually against the public datasets before being
considered complete, and every design decision was discussed and understood
before implementation, not accepted blindly.


## System Architecture

The system is organized in layers that mirror the four RAG stages, with a
deliberate separation between **orchestration** (what to do and in what order) and **strategy** (how a given stage is actually implemented). This separation is what allowed the five bonus features to be added without reworking the mandatory pipeline.

### Layers

- **CLI (`src/__main__.py`)**: the single entry point, built with Python Fire.

- **Local HTTP API (`src/api/app.py`, bonus)**: a thin FastAPI wrapper around the exact same services used by the CLI. No logic is duplicated.

- **Indexing (`src/indexer.py`, `src/chunking/`)**: discovers files, reads
  them, routes them to the appropriate chunker (`CodeChunker` or
  `TextChunker`, both implementing a shared `BaseChunker` interface), builds a BM25 index, and persists it to disk.

- **Retrieval (`src/retrieval/`, `src/retriever_service.py`)**: all retrieval
  strategies (`LexicalRetriever`, `SemanticRetriever`, `HybridRetriever`,
  `CachedRetriever`) implement a shared `BaseRetriever` interface
  (`search(query, k) -> list[ScoredChunk]`). `RetrieverService` is the
  orchestrator: it does not know or care which strategy is behind
  `BaseRetriever`, it only converts results into the pydantic models required
  for output.

- **Generation (`src/generation/`, `src/generator_service.py`)**: builds the
  prompt from retrieved sources, calls Qwen3-0.6B, and returns a grounded
  answer. `GeneratorService` plays the same orchestrator role as
  `RetrieverService`.

- **Evaluation (`src/evaluator.py`)**: computes recall@k by comparing
  student-produced search results against a ground-truth dataset, using an
  IoU-based matching rule.

### Why this separation matters

Because every retrieval strategy implements the same `BaseRetriever`
interface, `RetrieverService`, the CLI, and the HTTP API never had to change
when semantic embeddings, hybrid retrieval, or caching were added — only new
classes were created, and one factory function (`_build_retriever`) decides
which one to instantiate based on a `--retrieval_mode` flag.

## Bonus Features

All bonuses are opt-in: the mandatory pipeline (`index`, `search`, `answer`, ...)
works without any flag. Semantic embeddings and hybrid retrieval are described
in the Retrieval Method section.

### Caching

Two levels of caching are implemented in `src/caching/cache.py`:

- **Index caching**: `get_cached_lexical_retriever` is decorated with
  `functools.lru_cache`, so the lexical index is loaded from disk once per
  process and reused afterwards.
- **Query caching**: `CachedRetriever` wraps any retriever (lexical, semantic
  or hybrid) and stores results per `(query, k)`. It keeps at most `maxsize`
  entries (128 by default) and evicts the oldest entry when full.

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5 --use_cache True
```

Because every CLI command is a new process, the query cache only pays off in a
long-running process: repeated questions sent to the HTTP API are answered
from memory instead of searching again.

### Incremental indexing

`index_incremental` re-chunks only the files that are new or changed since the
last run, instead of reading and chunking the whole corpus.

1. A `manifest.json` in `data/processed/` stores the MD5 hash of the content
   of every indexed file.
2. On the next run, each file is hashed again. If its hash is unchanged, its
   chunks are reused from `chunks.pkl`. Otherwise (or if the file is new) it
   is chunked again.
3. Files that no longer exist are dropped from the index.
4. The BM25 index is rebuilt from all chunks (reused and new) and saved.

Hashing the content instead of using the modification time means that
`touch file.py` does not trigger re-indexing.

```bash
uv run python -m src index                 # full indexing, creates the first index
uv run python -m src index_incremental     # "0 changed/new files, N unchanged"
# edit one file under data/raw/, then:
uv run python -m src index_incremental     # "1 changed/new files, N-1 unchanged"
```

Limitation: the BM25 index itself is rebuilt on each run (only the reading and
chunking are skipped), and the semantic index is not updated incrementally.
Run `index --build_semantic True` to refresh it.

### Local HTTP API

`serve` starts a FastAPI server (`src/api/app.py`) that exposes the same
services as the CLI (`RetrieverService` and `GeneratorService`), without
duplicating any logic. The index and the Qwen model are loaded once at
startup. The API uses the lexical retriever with the query cache.

```bash
uv run python -m src serve --host 127.0.0.1 --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Returns `{"status": "ok"}` |
| `GET /search?query=...&k=5` | Top-k sources (`MinimalSearchResults`) |
| `GET /answer?query=...&k=5` | Generated answer with its sources (`MinimalAnswer`) |

```bash
curl "http://127.0.0.1:8000/search?query=How%20to%20configure%20the%20OpenAI%20server&k=3"
curl "http://127.0.0.1:8000/answer?query=How%20to%20configure%20the%20OpenAI%20server&k=3"
```

Errors never return a traceback: an empty query or a negative `k` returns
`400`, and `503` is returned if the index has not been built yet. Interactive
documentation is available at `http://127.0.0.1:8000/docs`.

## Chunking Strategy

Python code and Markdown documentation do not break apart the same way, so two
distinct chunking strategies were implemented, both built on
`langchain_text_splitters.RecursiveCharacterTextSplitter`:

- **`CodeChunker`** uses `Language.PYTHON` separators, so the splitter tries to
  break at logical boundaries (class/function definitions, blank lines)
  before falling back to splitting mid-line. This avoids cutting a function
  or class in half whenever possible.

- **`TextChunker`** uses `Language.MARKDOWN` separators, so it prefers to
  split at header boundaries (`#`, `##`, ...) before falling back to
  paragraph or line breaks, keeping each chunk close to one logical section
  of a document.

Both chunkers share the same interface (`BaseChunker.chunk(content, file_path,
max_chunk_size) -> list[Chunk]`) and the same `Chunk` data structure
(`text`, `file_path`, `first_character_index`, `last_character_index`), using
`add_start_index=True` on the splitter to recover the exact character offsets
of each chunk in the original file — this is what guarantees the
`first_character_index` / `last_character_index` reported to the moulinette
are accurate.

### Configuration

- `max_chunk_size` is a configurable CLI argument (`--max_chunk_size`),
  defaulting to **2000 characters** as required by the subject (the
  moulinette rejects any retrieved source longer than this).
- `chunk_overlap=0`: no overlap between consecutive chunks. This was kept
  simple for the mandatory part; a small overlap (100–200 characters) could
  be explored to reduce information loss at chunk boundaries, but was not
  found to be necessary given the recall@5 results obtained.

### Results on the vLLM 0.10.1 corpus

With `max_chunk_size=2000`, indexing the full corpus (1965 files) produced
**14,628 chunks** (≈13,800 from Python files, ≈800 from Markdown files) in
under 5 seconds — well within the 5-minute limit required by the subject.

## Retrieval Method

### Lexical retrieval (mandatory)

The mandatory retrieval strategy is **BM25**, implemented with the
[`bm25s`](https://github.com/xhluca/bm25s) library (chosen over `rank_bm25`
for its significantly faster indexing and query throughput, and its built-in
`.save()`/`.load()` persistence). Both the corpus chunks and incoming queries
are tokenized the same way (`bm25s.tokenize(..., stopwords="en")`), which is
essential: any mismatch between indexing-time and query-time tokenization
would silently degrade recall.

All retrieval strategies share a common interface,
`BaseRetriever.search(query, k) -> list[ScoredChunk]`, so the CLI, the HTTP
API, and `RetrieverService` never need to know which concrete strategy is
behind it.

### Semantic retrieval (bonus)

`SemanticRetriever` builds a dense vector index using
[`sentence-transformers`](https://www.sbert.net/) with the lightweight
`all-MiniLM-L6-v2` model. Each chunk is embedded once at index time; at query
time, the question is embedded and compared against every chunk via cosine
similarity (computed as a dot product, since embeddings are L2-normalized).

### Hybrid retrieval (bonus)

`HybridRetriever` combines the lexical and semantic rankings into a single
result list. Because BM25 scores and cosine similarity scores live on
completely different scales, each ranking is first **min-max normalized to
[0, 1]** independently, then combined with a weighted sum:

`alpha` defaults to `0.5` (equal weight) and is configurable.

### Selecting a retrieval strategy

The retrieval strategy is selected per-command via `--retrieval_mode`
(`lexical` by default, or `semantic` / `hybrid`), both for single queries
(`search`, `answer`) and for dataset-level commands (`search_dataset`).

## Performance Analysis

### Recall@k (lexical retrieval, mandatory)

Measured with the project's own `evaluate` command against the public
ground-truth datasets (`data/datasets/AnsweredQuestions/`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Threshold (Recall@5) |
|---------|----------|----------|----------|-----------|------------------------|
| Docs    | 62.0%    | 79.0%    | **83.0%**  | 91.0%     | 80% ✓                  |
| Code    | 32.3%    | 45.5%    | **55.6%**  | 59.6%     | 50% ✓                  |

Both thresholds required by the subject are met, with a comfortable margin on
docs questions and a smaller but sufficient margin on code questions.

Code questions are consistently harder than docs questions across every k —
this is expected, since code questions often rely on exact identifiers
(function names, variable names, flags) that lexical matching handles less
robustly than natural-language phrasing found in documentation.

### Indexing and retrieval speed

| Metric                                   | Result                  | Requirement        |
|-------------------------------------------|--------------------------|---------------------|
| Indexing time (1965 files, 14,628 chunks) | < 5 seconds  (lixical)   | ≤ 5 minutes         |
| Search throughput (100 questions)       | < 1 second (~400–560 it/s) | ≤ 90s for 200 questions |

Both performance requirements are met with a very large margin.

### Generation time

Answer generation (Qwen3-0.6B, CPU, `max_new_tokens=400`, `enable_thinking=False`)
takes approximately **19 seconds per question** on the test machine. The
subject does not impose a time limit on generation (only on indexing and
retrieval), so this was not optimized further for the mandatory part.


## Design Decisions

- **`BaseChunker` / `BaseRetriever` abstractions**: every chunker and
  retriever implements a shared interface, so new strategies (semantic,
  hybrid, cached) could be added as pure additions, with zero changes to the
  CLI, the API, or the orchestrator services.
- **Service layer (`RetrieverService`, `GeneratorService`)**: separates
  *orchestration* (building pydantic models, batching, progress bars) from
  *strategy* (how search or generation actually works), keeping each layer
  independently testable.
- **`bm25s` over `rank_bm25`**: faster indexing/querying and built-in
  `.save()`/`.load()`, both useful given the 5-minute indexing and 90-second
  retrieval limits.
- **`max_sources=3` in generation**: limiting context to the top-3 retrieved
  chunks (not a token-limit concern — Qwen3's context window is large) keeps
  the prompt focused and generation fast on CPU.
- **Error handling pushed to the CLI layer**: low-level methods (`read_file`,
  `save_index`) raise exceptions rather than swallowing them, so the CLI
  decides what is recoverable (skip a bad file) versus fatal (stop on a
  failed save).

  ## Challenges Faced

- **Qwen3's thinking mode**: by default, Qwen3-0.6B wrapped answers in
  `<think>...</think>` blocks, sometimes truncating the actual answer. Fixed
  with `enable_thinking=False` in `apply_chat_template`, plus a defensive
  fallback that strips any leftover `<think>` block.
- **Score scale mismatch in hybrid retrieval**: BM25 scores and cosine
  similarity scores are not comparable directly. Solved with independent
  min-max normalization before combining them.
- **`file_path` must match the corpus path exactly**: since the moulinette
  compares paths verbatim, care was taken to keep `raw_dir` relative so that
  every `Chunk.file_path` naturally matches the expected format
  (`data/raw/vllm-0.10.1/...`).

  ## Example Usage

```bash
# Build the indexF
uv run python3 -m src index --max_chunk_size 2000

# Ask a single question
uv run python3 -m src answer "How to configure the OpenAI server?" --k 5

# Run the HTTP API (bonus)
uv run python3 -m src serve
curl "http://127.0.0.1:8000/answer?query=How%20to%20configure%20the%20OpenAI%20server&k=3"
```