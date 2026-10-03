from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


@dataclass(frozen=True)
class SourceConfig:
    location: str
    recursive: bool
    max_depth: int = 2


@dataclass(frozen=True)
class VectorStoreConfig:
    directory: Path
    collection: str


@dataclass(frozen=True)
class ChunkingConfig:
    size: int
    overlap: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 4


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    chat: str
    embedding: str


@dataclass(frozen=True)
class AppConfig:
    source: SourceConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: ModelConfig


@dataclass(frozen=True)
class LoadReport:
    documents: list[Document]
    skipped: int = 0
    failures: list[str] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            object.__setattr__(self, "failures", [])


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{name}' section")
    return value


def _required(section: dict, name: str, section_name: str):
    if name not in section:
        raise ValueError(f"Missing '{section_name}.{name}'")
    return section[name]


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read config '{config_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config must contain a YAML mapping")

    source_data = _section(data, "source")
    vector_data = _section(data, "vector_store")
    chunk_data = _section(data, "chunking")
    model_data = _section(data, "models")
    retrieval_data = data.get("retrieval", {})
    if not isinstance(retrieval_data, dict):
        raise ValueError("Invalid 'retrieval' section")

    location = str(_required(source_data, "location", "source"))
    if not location.startswith(("http://", "https://")):
        location = str((config_path.parent / location).resolve())
    max_depth = source_data.get("max_depth", 2)
    recursive = source_data.get("recursive", False)
    size = _required(chunk_data, "size", "chunking")
    overlap = _required(chunk_data, "overlap", "chunking")
    top_k = retrieval_data.get("top_k", 4)

    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("source.max_depth must be a non-negative integer")
    if not isinstance(recursive, bool):
        raise ValueError("source.recursive must be true or false")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("chunking.size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= size:
        raise ValueError("chunking.overlap must be non-negative and smaller than size")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("retrieval.top_k must be a positive integer")

    directory = Path(str(_required(vector_data, "directory", "vector_store")))
    if not directory.is_absolute():
        directory = (config_path.parent / directory).resolve()

    return AppConfig(
        source=SourceConfig(location, recursive, max_depth),
        vector_store=VectorStoreConfig(
            directory, str(_required(vector_data, "collection", "vector_store"))
        ),
        chunking=ChunkingConfig(size, overlap),
        retrieval=RetrievalConfig(top_k),
        models=ModelConfig(
            str(_required(model_data, "provider", "models")),
            str(_required(model_data, "chat", "models")),
            str(_required(model_data, "embedding", "models")),
        ),
    )


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _default_fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Pasha-RAG/1.0"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8")


def _load_file(path: Path) -> list[Document]:
    resolved = path.resolve()
    if path.suffix.lower() == ".pdf":
        documents = PyPDFLoader(str(resolved)).load()
    else:
        documents = TextLoader(str(resolved), encoding="utf-8").load()
    for document in documents:
        document.metadata["source"] = str(resolved)
    return documents


def _load_local(source: SourceConfig) -> LoadReport:
    path = Path(source.location)
    if not path.exists():
        raise ValueError(f"Source path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file: {path}")
        return LoadReport(_load_file(path))

    candidates = sorted(path.rglob("*") if source.recursive else path.glob("*"))
    files = [candidate for candidate in candidates if candidate.is_file()]
    supported = [file for file in files if file.suffix.lower() in SUPPORTED_EXTENSIONS]
    documents = [document for file in supported for document in _load_file(file)]
    return LoadReport(documents, skipped=len(files) - len(supported))


def _load_web(source: SourceConfig, fetch: Callable[[str], str]) -> LoadReport:
    start = normalize_url(source.location)
    hostname = urlsplit(start).hostname
    queue = deque([(start, 0)])
    visited: set[str] = set()
    documents: list[Document] = []
    failures: list[str] = []

    while queue:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            html = fetch(url)
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        documents.append(Document(page_content=soup.get_text(" ", strip=True), metadata={"source": url}))

        if not source.recursive or depth >= source.max_depth:
            continue
        for anchor in soup.find_all("a", href=True):
            child = normalize_url(urljoin(url, anchor["href"]))
            parsed = urlsplit(child)
            if parsed.scheme in {"http", "https"} and parsed.hostname == hostname and child not in visited:
                queue.append((child, depth + 1))

    return LoadReport(documents, failures=failures)


def load_documents(
    config: AppConfig, fetch: Callable[[str], str] | None = None
) -> LoadReport:
    if config.source.location.startswith(("http://", "https://")):
        return _load_web(config.source, fetch or _default_fetch)
    return _load_local(config.source)
