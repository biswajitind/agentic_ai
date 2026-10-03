# Pasha RAG Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compact LangChain RAG application that ingests one configured website, file, or directory into local Chroma and serves answers through Python, Streamlit, and Telegram.

**Architecture:** Focused top-level modules share one validated YAML configuration and one persistent Chroma collection. `sources.py` owns source discovery/loading, `ingest.py` owns chunk lifecycle, `models.py` isolates OpenAI construction, and both interfaces delegate all retrieval and generation to `RAGAgent`.

**Tech Stack:** Python 3.11+, LangChain, LangChain OpenAI, LangChain Chroma, ChromaDB, Streamlit, python-telegram-bot, Beautiful Soup, PyPDF, PyYAML, python-dotenv, pytest, uv

**Spec:** `Pasha/docs/superpowers/specs/2026-10-03-rag-application-design.md`

## Global Constraints

- Keep production code small and readable; do not introduce packaging metadata or framework abstractions for version one.
- OpenAI is the only implemented provider, but model construction must remain isolated in `models.py`.
- Support HTTP(S) pages and local `.pdf`, `.md`, and `.txt` files only.
- Recursive web crawling is exact-host only, cycle-safe, and defaults to depth 2; the starting page is depth 0.
- Resolve relative source and Chroma paths against the configuration file's directory.
- Persist vectors locally in Chroma and make repeat ingestion idempotent, including cleanup of stale chunks for changed sources.
- Load secrets from `.env`; never place or log secrets in YAML.
- Tests must not make live internet, OpenAI, or Telegram calls.
- Use TDD for every behavior-bearing task and commit after each task passes.

## Review Focus

- URLs that differ only by fragments or trailing slashes must not be crawled repeatedly; pinned in Task 1 URL-normalization tests.
- A recursive crawl with one failed page must retain successful pages and report the failure; pinned in Task 1 partial-failure test.
- Re-ingesting edited content must remove stale chunks for that source without deleting other sources; pinned in Task 3 lifecycle test.
- A Chroma directory may exist while the configured collection is empty; pinned in Task 4 empty-collection test.
- Telegram answers may exceed its message limit; Task 5 must split replies into chunks no longer than 4096 characters.

---

### Task 1: Configuration and Source Loading

**Files:**
- Create: `Pasha/sources.py`
- Create: `Pasha/config.yaml`
- Create: `Pasha/tests/test_config.py`
- Create: `Pasha/tests/test_sources.py`

**Interfaces:**
- Consumes: YAML path and local/network source locations.
- Produces: `SourceConfig(location: str, recursive: bool, max_depth: int)`, `VectorStoreConfig(directory: Path, collection: str)`, `ChunkingConfig(size: int, overlap: int)`, `RetrievalConfig(top_k: int)`, `ModelConfig(provider: str, chat: str, embedding: str)`, and aggregate `AppConfig`; `load_config(path: str | Path) -> AppConfig`; `normalize_url(url: str) -> str`; `load_documents(config: AppConfig, fetch: Callable[[str], str] | None = None) -> LoadReport` where `LoadReport` exposes `documents: list[Document]`, `skipped: int`, and `failures: list[str]`.

- [ ] **Step 1: Write failing configuration tests**

Add tests asserting that `load_config()` applies `max_depth=2` and `top_k=4`, resolves relative paths against the YAML file, accepts the sample config, rejects missing sections, rejects `overlap >= size`, and rejects negative crawl depth with field-specific `ValueError` messages. Preserve the configured provider string so `models.py` owns provider support errors.

- [ ] **Step 2: Run configuration tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_config.py -v`

Expected: FAIL because `sources.py` and its configuration types do not exist.

- [ ] **Step 3: Implement configuration types and `load_config()`**

Use frozen standard-library dataclasses and `yaml.safe_load`; keep defaults limited to the exact values in the spec. Create the agreed sample `config.yaml` with a non-recursive `./documents` source so a first run cannot unexpectedly crawl.

- [ ] **Step 4: Run configuration tests and verify pass**

Run: `cd Pasha && uv run pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing local and web source tests**

Cover one supported file, explicit unsupported-file error, recursive/non-recursive directory discovery, skipped unsupported files, PDF page metadata, exact-host filtering, relative-link resolution, fragment/trailing-slash normalization, cycle prevention, depth 0 versus depth 2, and a failed child page that leaves successful documents in `LoadReport`.

- [ ] **Step 6: Run source tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_sources.py -v`

Expected: FAIL because source discovery and crawling functions are missing.

- [ ] **Step 7: Implement source loading**

Use LangChain `TextLoader` for `.txt`/`.md`, `PyPDFLoader` for PDF, `urllib.request` for HTTP, Beautiful Soup for visible text and links, and breadth-first `(url, depth)` traversal. Accept an injected `fetch` callable in tests; the default fetcher supplies a descriptive user agent and timeout.

- [ ] **Step 8: Run Task 1 tests**

Run: `cd Pasha && uv run pytest tests/test_config.py tests/test_sources.py -v`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add Pasha/config.yaml Pasha/sources.py Pasha/tests/test_config.py Pasha/tests/test_sources.py
git commit -m "feat: add configured document loading"
```

### Task 2: Configurable Model Factory

**Files:**
- Create: `Pasha/models.py`
- Create: `Pasha/tests/test_models.py`

**Interfaces:**
- Consumes: `ModelConfig` from Task 1 and `OPENAI_API_KEY` from the environment.
- Produces: `create_chat_model(config: ModelConfig) -> BaseChatModel`; `create_embedding_model(config: ModelConfig) -> Embeddings`.

- [ ] **Step 1: Write failing model-factory tests**

Mock `ChatOpenAI` and `OpenAIEmbeddings`; assert the configured model names are forwarded, missing `OPENAI_API_KEY` raises an actionable error, and unsupported providers cannot silently fall back to OpenAI.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_models.py -v`

Expected: FAIL because `models.py` does not exist.

- [ ] **Step 3: Implement the two factory functions**

Read the key from the environment only to validate its presence; allow LangChain's OpenAI classes to consume it normally. Keep provider selection in one private guard so adding a provider changes only this module.

- [ ] **Step 4: Run Task 2 tests**

Run: `cd Pasha && uv run pytest tests/test_models.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add Pasha/models.py Pasha/tests/test_models.py
git commit -m "feat: add configurable model factories"
```

### Task 3: Idempotent Chroma Ingestion CLI

**Files:**
- Create: `Pasha/ingest.py`
- Create: `Pasha/tests/test_ingest.py`

**Interfaces:**
- Consumes: `AppConfig`, `LoadReport`, and `create_embedding_model()`.
- Produces: `split_documents(documents: list[Document], config: ChunkingConfig) -> list[Document]`; `chunk_id(document: Document) -> str`; `ingest(config: AppConfig, reset: bool = False) -> IngestReport`; `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing pure ingestion tests**

Assert configured chunk size/overlap is applied, the same source/page/content yields the same SHA-256 ID, changed content yields a different ID, and missing source metadata is rejected before storage.

- [ ] **Step 2: Run pure tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_ingest.py -v`

Expected: FAIL because `ingest.py` does not exist.

- [ ] **Step 3: Implement splitting and deterministic IDs**

Use `RecursiveCharacterTextSplitter`; hash a UTF-8 serialization of normalized `source`, optional `page`, and `page_content`.

- [ ] **Step 4: Add failing Chroma lifecycle and CLI tests**

Using a temporary persistent Chroma directory and fake deterministic embeddings, assert first ingestion writes chunks, identical ingestion does not increase the count, edited-source ingestion removes stale IDs, a second source remains untouched, `reset=True` recreates only the configured collection, empty loads fail, and `main()` parses `--config`/`--reset` and prints all summary counts.

- [ ] **Step 5: Run lifecycle tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_ingest.py -v`

Expected: FAIL on the unimplemented storage lifecycle.

- [ ] **Step 6: Implement ingestion and CLI**

Construct `langchain_chroma.Chroma` with the configured directory and collection. For every normalized source represented in the new chunks, query existing records by source metadata, delete IDs absent from the new set, then call `add_documents(..., ids=...)`. `--reset` uses the Chroma collection deletion API for the named collection only. Call `load_dotenv()` in `main()` before loading models.

- [ ] **Step 7: Run Task 3 tests**

Run: `cd Pasha && uv run pytest tests/test_ingest.py -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add Pasha/ingest.py Pasha/tests/test_ingest.py
git commit -m "feat: ingest documents into local Chroma"
```

### Task 4: Reusable RAG Agent

**Files:**
- Create: `Pasha/rag_agent.py`
- Create: `Pasha/tests/test_rag_agent.py`

**Interfaces:**
- Consumes: `AppConfig`, `load_config()`, both model factories, and the configured Chroma collection.
- Produces: `RAGAgent.from_config(path: str | Path) -> RAGAgent`; `RAGAgent.ask(question: str) -> dict[str, object]` with keys `answer` and `sources`.

- [ ] **Step 1: Write failing agent tests**

Inject a fake retriever and fake chat model. Assert blank questions are rejected; `top_k` is forwarded; retrieved text and source labels appear in the prompt; no retrieved documents returns the exact answer `I don't know based on the available documents.` without calling the model; response content is normalized to a string; source/page dictionaries are deduplicated in retrieval order; and an existing but empty Chroma collection produces the run-ingestion error.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_rag_agent.py -v`

Expected: FAIL because `rag_agent.py` does not exist.

- [ ] **Step 3: Implement `RAGAgent`**

Allow constructor injection of retriever and chat model for tests. `from_config()` creates embeddings, Chroma, validates collection count, builds a `similarity` retriever with `k=top_k`, and creates the chat model. Build the grounding instruction and retrieved context with LangChain messages, then call `invoke()`.

- [ ] **Step 4: Run Task 4 tests**

Run: `cd Pasha && uv run pytest tests/test_rag_agent.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add Pasha/rag_agent.py Pasha/tests/test_rag_agent.py
git commit -m "feat: add reusable RAG agent"
```

### Task 5: Streamlit and Telegram Adapters

**Files:**
- Create: `Pasha/app.py`
- Create: `Pasha/telegram_bot.py`
- Create: `Pasha/tests/test_interfaces.py`

**Interfaces:**
- Consumes: `RAGAgent.from_config()`, `.env`, and optional `PASHA_CONFIG`.
- Produces: Streamlit entry module; Telegram `start(update, context)`, `answer(update, context)`, `split_message(text: str, limit: int = 4096) -> list[str]`, and `main() -> None`.

- [ ] **Step 1: Write failing interface tests**

Assert `split_message()` preserves all text and limits chunks to 4096 characters; `/start` sends usage text; ordinary Telegram text calls `ask()` and appends unique source labels; missing `TELEGRAM_BOT_TOKEN` names the variable; `PASHA_CONFIG` overrides `config.yaml`; and importing the Streamlit module with mocked Streamlit/RAG dependencies does not start ingestion or Telegram.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd Pasha && uv run pytest tests/test_interfaces.py -v`

Expected: FAIL because the interface modules do not exist.

- [ ] **Step 3: Implement the Streamlit adapter**

Call `load_dotenv()`, cache one agent using `st.cache_resource`, retain user/assistant messages in `st.session_state`, render `st.chat_input`, and show each answer's source dictionaries in an expander.

- [ ] **Step 4: Implement the Telegram adapter**

Call `load_dotenv()`, validate the token, build one `Application`, register `/start` and text handlers, and keep the agent in `application.bot_data`. Split oversized replies at newline or whitespace boundaries where possible and log exceptions without configuration or token values.

- [ ] **Step 5: Run Task 5 tests**

Run: `cd Pasha && uv run pytest tests/test_interfaces.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add Pasha/app.py Pasha/telegram_bot.py Pasha/tests/test_interfaces.py
git commit -m "feat: add Streamlit and Telegram interfaces"
```

### Task 6: Installation Files, Documentation, and End-to-End Verification

**Files:**
- Create: `Pasha/requirements.txt`
- Create: `Pasha/.env.example`
- Create: `Pasha/.gitignore`
- Create: `Pasha/readme.txt`
- Modify: `Pasha/tests/test_interfaces.py`

**Interfaces:**
- Consumes: all commands and configuration defined by Tasks 1-5.
- Produces: reproducible `uv` setup instructions and safe local defaults.

- [ ] **Step 1: Write the direct dependency ranges and validate resolution**

List only the direct packages named in the spec, including `langchain-text-splitters`, and select current mutually compatible bounded ranges using official package metadata. Create a clean temporary uv environment and run `uv pip install -r requirements.txt`; do not commit a local virtual environment.

Expected: dependency resolution and installation succeed on Python 3.11 or newer.

- [ ] **Step 2: Add `.env.example` and project `.gitignore`**

Document empty `OPENAI_API_KEY` and `TELEGRAM_BOT_TOKEN` values. Ignore `.env`, `data/`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, and `.streamlit/` without ignoring either example or documentation files.

- [ ] **Step 3: Write `readme.txt`**

Cover every documentation item in the spec. Use these primary commands verbatim:

```text
uv venv
uv pip install -r requirements.txt
uv run python ingest.py --config config.yaml
uv run python ingest.py --config config.yaml --reset
uv run streamlit run app.py
uv run python telegram_bot.py
uv run pytest
```

Include concrete YAML examples for one page, recursive website, one file, and recursive directory, plus `.env`, provider-extension, safe-reset, and troubleshooting instructions.

- [ ] **Step 4: Add and run import/CLI smoke tests**

Extend `tests/test_interfaces.py` to import every production module and invoke `ingest.py --help` in a subprocess. Verify the help command exits 0 without credentials, Streamlit/Telegram imports cause no network calls, and missing configuration/secret failures remain actionable.

Run: `cd Pasha && uv run pytest -v`

Expected: all tests PASS with no live service calls.

- [ ] **Step 5: Run static sanity checks**

Run: `cd Pasha && uv run python -m compileall -q . && uv run python ingest.py --help`

Expected: compilation exits 0 and CLI help lists `--config` and `--reset`.

- [ ] **Step 6: Review project size and documentation accuracy**

Confirm there is no duplicated RAG logic in the interfaces, no secret in tracked files, no unnecessary package metadata, and every command/path in `readme.txt` matches the implemented CLI.

- [ ] **Step 7: Commit Task 6**

```bash
git add Pasha/requirements.txt Pasha/.env.example Pasha/.gitignore Pasha/readme.txt Pasha/tests/test_interfaces.py
git commit -m "docs: add Pasha setup and usage guide"
```

### Task 7: Final Acceptance

**Files:**
- Modify only if verification exposes a defect.

**Interfaces:**
- Consumes: the complete application.
- Produces: verified release-ready project state.

- [ ] **Step 1: Run the full offline test suite from a clean environment**

Run: `cd Pasha && uv run pytest -v`

Expected: all tests PASS.

- [ ] **Step 2: Exercise local text ingestion with fake embeddings in a temporary Chroma directory**

Use the test seam from Task 3 rather than OpenAI. Ingest two small text files twice and assert the collection count is unchanged on the second run.

Expected: the second run reports the same stored chunk count with no duplicates.

- [ ] **Step 3: Run final repository checks**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intentional project changes, if any, are present.

- [ ] **Step 4: Commit any verification-only fixes**

If Step 1-3 required changes, add only the affected `Pasha/` files and commit with a message describing the verified defect. Otherwise, do not create an empty commit.
