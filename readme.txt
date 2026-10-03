Pasha RAG
=========

Pasha ingests one website, file, or directory into a local Chroma vector
store, then answers questions through Python, Streamlit, or Telegram.
Supported files are PDF, Markdown, and plain text. Python 3.11 or newer and
uv are required.

SETUP

From this directory, create a virtual environment. If your default Python is
already 3.11 or newer, this shorter command is sufficient:

    uv venv

To select the supported interpreter explicitly:

    uv venv --python 3.11

Install dependencies:

    uv pip install -r requirements.txt

Copy .env.example to .env and fill in the keys you use:

    OPENAI_API_KEY=your-openai-key
    TELEGRAM_BOT_TOKEN=your-telegram-token

OPENAI_API_KEY is required for ingestion and questions. TELEGRAM_BOT_TOKEN is
required only for the Telegram bot. Pasha loads .env with python-dotenv. Do
not commit .env.

CONFIGURATION

config.yaml contains these sections:

source
  location: A local file, local directory, or HTTP(S) URL. Relative local
            paths are resolved from the directory containing config.yaml.
  recursive: For a directory, include subdirectories. For a URL, follow
             links. For a file or single page, use false.
  max_depth: Maximum URL link depth. The starting page is depth 0. Recursive
             crawling remains on the exact starting hostname. Default: 2.

vector_store
  directory: Local Chroma persistence directory, relative to config.yaml.
  collection: Chroma collection name shared by ingestion and both interfaces.

chunking
  size: Maximum chunk size in characters. Must be positive.
  overlap: Repeated characters between chunks. Must be smaller than size.

retrieval
  top_k: Number of chunks supplied to the model. Default: 4.

models
  provider: openai in this version.
  chat: OpenAI chat model name.
  embedding: OpenAI embedding model name.

SOURCE EXAMPLES

One web page:

    source:
      location: "https://example.com/guide"
      recursive: false
      max_depth: 2

Recursive website, restricted to the starting hostname:

    source:
      location: "https://docs.example.com"
      recursive: true
      max_depth: 2

One local file:

    source:
      location: "./documents/guide.pdf"
      recursive: false
      max_depth: 2

Directory including subfolders:

    source:
      location: "./documents"
      recursive: true
      max_depth: 2

INGESTION

Run normal idempotent ingestion:

    uv run python ingest.py --config config.yaml

Rebuild only the collection named in config.yaml:

    uv run python ingest.py --config config.yaml --reset

Normal ingestion uses deterministic chunk IDs, updates changed source chunks,
and leaves chunks belonging to other sources intact. --reset deletes and
recreates the configured collection only; it does not delete other collections
or the whole Chroma directory.

STREAMLIT

    uv run streamlit run app.py

Open the URL printed by Streamlit and enter a question. Answers show expandable
source references. Set PASHA_CONFIG to use a config file other than config.yaml.

TELEGRAM

Create a bot with BotFather, put its token in TELEGRAM_BOT_TOKEN, then run:

    uv run python telegram_bot.py

Send /start or a normal text question. PASHA_CONFIG also selects an alternate
configuration for the bot.

TESTS

Tests use temporary storage and doubles; they do not call OpenAI, Telegram, or
the public internet.

    uv run pytest

ADDING ANOTHER MODEL PROVIDER

Install that provider's LangChain integration, add one branch to the validation
and two factory functions in models.py, and use its provider name in
config.yaml. Ingestion, RAGAgent, Streamlit, and Telegram need no changes.

TROUBLESHOOTING

"OPENAI_API_KEY is required"
  Add OPENAI_API_KEY to .env and rerun the command.

"TELEGRAM_BOT_TOKEN is required"
  Add TELEGRAM_BOT_TOKEN to .env before starting telegram_bot.py.

"run ingestion first"
  The configured Chroma collection is empty. Run the ingestion command using
  the same config file as the application.

Missing source path or unsupported file
  Check source.location. Supported local suffixes are .pdf, .md, and .txt.

Web failures
  Confirm network access and the URL. Recursive loading records failed pages
  while retaining pages downloaded successfully. JavaScript rendering and
  robots.txt processing are outside this version's scope.

Wrong collection or stale local data
  Confirm vector_store.directory and collection are shared by ingestion and
  the interface. Use --reset when an intentional full collection rebuild is
  required.
