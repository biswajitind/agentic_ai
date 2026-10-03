from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import create_embedding_model
from sources import AppConfig, ChunkingConfig, load_config, load_documents


@dataclass(frozen=True)
class IngestReport:
    documents_loaded: int
    chunks_stored: int
    skipped: int
    failures: int


def split_documents(
    documents: list[Document], config: ChunkingConfig
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size,
        chunk_overlap=config.overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def chunk_id(document: Document) -> str:
    source = document.metadata.get("source")
    if not source:
        raise ValueError("Document metadata must include source")
    identity = json.dumps(
        {
            "source": str(source),
            "page": document.metadata.get("page"),
            "content": document.page_content,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _store(config: AppConfig, embeddings) -> Chroma:
    return Chroma(
        collection_name=config.vector_store.collection,
        persist_directory=str(config.vector_store.directory),
        embedding_function=embeddings,
    )


def ingest(config: AppConfig, reset: bool = False) -> IngestReport:
    loaded = load_documents(config)
    if not loaded.documents:
        raise ValueError("No documents were loaded; check the configured source")

    chunks = split_documents(loaded.documents, config.chunking)
    ids = [chunk_id(chunk) for chunk in chunks]
    embeddings = create_embedding_model(config.models)
    store = _store(config, embeddings)
    if reset:
        store.delete_collection()
        store = _store(config, embeddings)

    ids_by_source: dict[str, set[str]] = {}
    for identifier, chunk in zip(ids, chunks):
        ids_by_source.setdefault(str(chunk.metadata["source"]), set()).add(identifier)

    for source, current_ids in ids_by_source.items():
        existing = store.get(where={"source": source}, include=[])["ids"]
        stale = [identifier for identifier in existing if identifier not in current_ids]
        if stale:
            store.delete(ids=stale)

    store.add_documents(chunks, ids=ids)
    return IngestReport(
        documents_loaded=len(loaded.documents),
        chunks_stored=len(chunks),
        skipped=loaded.skipped,
        failures=len(loaded.failures or []),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Pasha's Chroma store")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--reset", action="store_true", help="Recreate the configured collection")
    args = parser.parse_args(argv)

    load_dotenv()
    report = ingest(load_config(args.config), reset=args.reset)
    print(f"Documents loaded: {report.documents_loaded}")
    print(f"Chunks stored: {report.chunks_stored}")
    print(f"Unsupported files skipped: {report.skipped}")
    print(f"Load failures: {report.failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
