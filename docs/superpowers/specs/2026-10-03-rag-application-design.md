# Pasha RAG Application Design

## Purpose

Pasha is a small local retrieval-augmented generation application. It ingests a configured website, file, or directory into a persistent Chroma collection and exposes the resulting knowledge base through a reusable Python agent, a Streamlit chat page, and a Telegram bot.

The first release prioritizes readable code, few abstractions, and a small direct dependency set. OpenAI is the supported model provider. Model construction is isolated so a later LangChain-supported provider can be added without changing ingestion, retrieval, Streamlit, or Telegram code.

## Success Criteria

- One configuration file selects a URL, file, or directory for an ingestion run.
- URL and directory recursion can be enabled or disabled.
- Recursive URL ingestion stays on the starting domain and stops at a configurable depth that defaults to 2.
- Local ingestion supports PDF, Markdown, and plain-text files.
- Repeating ingestion does not create duplicate chunks.
- Chroma data persists locally and is shared by all interfaces.
- A reusable `RAGAgent` answers from retrieved context and returns source references.
- Streamlit and Telegram provide thin interfaces over the same agent.
- Secrets load from `.env` with `python-dotenv` and never appear in YAML.
- The project includes automated tests and complete `uv`-based setup and usage instructions.

## Project Structure

```text
Pasha/
|-- app.py
|-- config.yaml
|-- ingest.py
|-- models.py
|-- rag_agent.py
|-- sources.py
|-- telegram_bot.py
|-- requirements.txt
|-- readme.txt
|-- .env.example
|-- .gitignore
|-- tests/
|   |-- test_config.py
|   |-- test_sources.py
|   |-- test_ingest.py
|   `-- test_rag_agent.py
`-- docs/superpowers/
    |-- specs/
    `-- plans/
```

Each production file has one responsibility:

- `sources.py` parses YAML configuration, discovers source documents, crawls permitted links, and returns LangChain `Document` objects.
- `models.py` constructs the configured chat and embedding models.
- `ingest.py` is the ingestion CLI. It splits documents, creates deterministic chunk IDs, and writes them to Chroma.
- `rag_agent.py` contains the reusable retrieval and answer-generation class.
- `app.py` is the Streamlit adapter.
- `telegram_bot.py` is the Telegram adapter.

Configuration parsing remains in `sources.py` to avoid creating another production module solely for a small dataclass and loader.

## Configuration

`config.yaml` describes one source per ingestion run and the shared vector/model settings:

```yaml
source:
  location: "./documents"
  recursive: true
  max_depth: 2

vector_store:
  directory: "./data/chroma"
  collection: "pasha"

chunking:
  size: 1000
  overlap: 150

retrieval:
  top_k: 4

models:
  provider: "openai"
  chat: "gpt-4.1-mini"
  embedding: "text-embedding-3-small"
```

Relative source and Chroma paths are resolved relative to the configuration file, not the caller's working directory. Required sections and values are validated on startup. Chunk size must be positive, overlap must be non-negative and smaller than chunk size, `top_k` must be positive, and `max_depth` must be a non-negative integer. An unsupported provider produces an actionable error from `models.py`.

OpenAI is the only implemented provider in version one. Adding a provider means adding its LangChain integration dependency and one provider branch in `models.py`; consumers continue to use the same chat-model and embedding-model factory functions.

## Source Loading and Crawling

The source location is classified as an HTTP(S) URL, local file, or local directory.

- A URL with recursion disabled loads only the configured page.
- A URL with recursion enabled follows normalized HTTP(S) links on the exact starting hostname through `max_depth`; the starting page is depth 0.
- Fragments are removed when normalizing URLs. Non-HTTP links, cross-domain links, and already-visited URLs are ignored.
- A file loads exactly one `.pdf`, `.md`, or `.txt` document. Selecting an unsupported file explicitly is an error.
- A non-recursive directory loads supported files immediately within that directory.
- A recursive directory loads supported files in all subdirectories.
- Unsupported files found during directory discovery are skipped and counted.

Page downloads occur only during ingestion. Retrieved content is converted into LangChain `Document` objects with source metadata. PDF documents also retain page-number metadata supplied by the loader. A crawl reports individual page failures and continues processing pages already discovered.

## Ingestion and Chroma Storage

The ingestion command is:

```bash
uv run python ingest.py --config config.yaml
```

Documents are split with LangChain's recursive character splitter using the configured size and overlap. Each chunk receives a deterministic identifier derived from its normalized source identity, page number when present, and chunk content. Chroma upserts chunks into the configured persistent collection, so processing identical content again does not multiply it.

If content at an existing source changes, new chunks are upserted. Before that source is written, the ingestion pipeline deletes older chunks whose IDs no longer occur in the newly produced set, preventing stale answers from edited documents. Documents from other sources remain untouched.

The command prints counts for source documents loaded, chunks stored, unsupported files skipped, and load failures. `--reset` deletes and recreates only the configured Chroma collection before ingestion. It does not delete the entire Chroma directory or unrelated collections.

## RAG Agent

The public interface is intentionally small:

```python
agent = RAGAgent.from_config("config.yaml")
result = agent.ask("What does the documentation say about deployment?")
```

`ask(question: str) -> dict[str, object]` returns:

```python
{
    "answer": "...",
    "sources": [
        {"source": "https://example.com/page", "page": None}
    ],
}
```

The agent connects to the configured persistent Chroma collection, retrieves `top_k` chunks, formats a compact context, and sends it to the configured chat model. The prompt instructs the model to answer only from supplied context and state that it does not know when the context is insufficient. Empty questions are rejected. Duplicate source references are removed while preserving retrieval order.

If the collection is absent or empty, construction or the first query raises a clear error directing the user to run ingestion. The agent does not fetch source web pages at query time.

## Streamlit and Telegram Interfaces

`app.py` calls `load_dotenv()`, creates one cached `RAGAgent`, and renders a title, Streamlit chat history, chat input, answers, and expandable source references. It does not contain ingestion or retrieval logic.

Launch it with:

```bash
uv run streamlit run app.py
```

`telegram_bot.py` calls `load_dotenv()`, reads `TELEGRAM_BOT_TOKEN`, creates one `RAGAgent`, handles `/start`, and passes ordinary text messages to `ask()`. Replies contain the answer followed by a compact, deduplicated source list. Telegram transport failures are logged without secrets or full configuration values.

Launch it with:

```bash
uv run python telegram_bot.py
```

Both interfaces use `config.yaml` by default and accept an optional configuration path through the `PASHA_CONFIG` environment variable. This avoids separate CLI parsing inside framework-controlled entry points.

## Secrets and Local Files

Every executable entry point calls `python-dotenv`'s `load_dotenv()` before constructing models or the Telegram application. `.env.example` documents:

```dotenv
OPENAI_API_KEY=
TELEGRAM_BOT_TOKEN=
```

`.gitignore` excludes `.env`, `data/`, virtual environments, test caches, and Python bytecode. API keys and bot tokens are never placed in `config.yaml` or logged.

## Errors and Observability

User-facing failures identify the corrective action:

- Missing or invalid YAML names the field and expected value.
- A missing `OPENAI_API_KEY` identifies that environment variable.
- Starting the Telegram bot without `TELEGRAM_BOT_TOKEN` identifies that variable.
- A missing local source path is an error.
- An explicitly selected unsupported file is an error.
- A completely unsuccessful load or an empty collection is an error rather than a successful no-op.
- Individual crawl failures are summarized while successful pages continue through ingestion.

Normal commands use concise console output. Python logging is reserved for crawl and Telegram runtime failures; secrets are never included.

## Dependencies

`requirements.txt` contains compatible version ranges for the direct dependencies only:

- LangChain core and community packages
- LangChain OpenAI integration
- LangChain Chroma integration and ChromaDB
- Streamlit
- Python Telegram Bot
- Beautiful Soup
- PyPDF
- PyYAML
- python-dotenv
- pytest

The implementation will select mutually compatible current ranges and record the minimum supported Python version in `readme.txt`. `uv` installs from `requirements.txt`; the project does not need packaging metadata for version one.

## Testing

Tests use `pytest`, temporary directories, and mocks. They make no live OpenAI, Telegram, or internet calls.

- Configuration tests cover relative-path resolution, defaults, required values, invalid chunk settings, invalid crawl depth, and unsupported providers.
- Source tests cover single files, directory recursion, supported extensions, same-host filtering, URL normalization, visited-link cycles, depth limits, and partial page failures.
- Ingestion tests cover splitting, deterministic IDs, repeat ingestion, changed-source cleanup, source isolation, and collection-only reset behavior.
- Agent tests cover retrieval prompt construction, insufficient context, empty questions, empty collections, result shape, and source deduplication.
- Interface-level smoke tests verify imports and the expected missing-configuration or missing-secret errors without starting servers.

The complete suite runs with:

```bash
uv run pytest
```

## Documentation

`readme.txt` explains:

- Python and `uv` prerequisites.
- Environment creation and dependency installation.
- Creating `.env` from `.env.example`.
- Every `config.yaml` field.
- Single-page, recursive-site, single-file, and recursive-directory examples.
- Ingestion, reset, Streamlit, Telegram, and test commands.
- Supported formats and same-domain crawl behavior.
- Local Chroma persistence and safe collection rebuilding.
- The small code change and dependency needed to add another LangChain provider.
- Common configuration, credential, collection, and network errors.

## Out of Scope

Version one does not include authentication, remote vector databases, scheduled ingestion, incremental crawl timestamps, robots.txt enforcement, JavaScript rendering, document upload through Streamlit or Telegram, conversation memory beyond Streamlit's visible UI history, multiple configured sources in one run, reranking, hybrid search, or a second implemented model provider.
