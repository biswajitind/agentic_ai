from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from sources import (
    AppConfig,
    ChunkingConfig,
    LoadReport,
    ModelConfig,
    RetrievalConfig,
    SourceConfig,
    VectorStoreConfig,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(sum(map(ord, text)) % 17), 1.0]


def app_config(tmp_path: Path, collection: str = "test") -> AppConfig:
    return AppConfig(
        source=SourceConfig(str(tmp_path / "docs"), True, 2),
        vector_store=VectorStoreConfig(tmp_path / "chroma", collection),
        chunking=ChunkingConfig(100, 10),
        retrieval=RetrievalConfig(4),
        models=ModelConfig("openai", "chat", "embed"),
    )


def test_split_documents_uses_configured_size_and_overlap():
    from ingest import split_documents

    chunks = split_documents(
        [Document(page_content="abcdefghij", metadata={"source": "notes.txt"})],
        ChunkingConfig(size=5, overlap=1),
    )

    assert [chunk.page_content for chunk in chunks] == ["abcde", "efghi", "ij"]
    assert all(chunk.metadata["source"] == "notes.txt" for chunk in chunks)


def test_chunk_id_is_stable_and_content_sensitive():
    from ingest import chunk_id

    first = Document(page_content="alpha", metadata={"source": "notes.txt", "page": 2})
    same = Document(page_content="alpha", metadata={"source": "notes.txt", "page": 2})
    changed = Document(page_content="beta", metadata={"source": "notes.txt", "page": 2})

    assert chunk_id(first) == chunk_id(same)
    assert chunk_id(first) != chunk_id(changed)
    assert len(chunk_id(first)) == 64


def test_chunk_id_rejects_missing_source_metadata():
    from ingest import chunk_id

    with pytest.raises(ValueError, match="source"):
        chunk_id(Document(page_content="orphan"))


def test_repeated_and_changed_ingestion_replaces_only_that_source(tmp_path, monkeypatch):
    import ingest

    config = app_config(tmp_path)
    reports = iter(
        [
            LoadReport(
                [
                    Document(page_content="old alpha", metadata={"source": "a.txt"}),
                    Document(page_content="bravo", metadata={"source": "b.txt"}),
                ]
            ),
            LoadReport(
                [
                    Document(page_content="old alpha", metadata={"source": "a.txt"}),
                    Document(page_content="bravo", metadata={"source": "b.txt"}),
                ]
            ),
            LoadReport(
                [
                    Document(page_content="new alpha", metadata={"source": "a.txt"}),
                    Document(page_content="bravo", metadata={"source": "b.txt"}),
                ]
            ),
        ]
    )
    monkeypatch.setattr(ingest, "load_documents", lambda _: next(reports))
    monkeypatch.setattr(ingest, "create_embedding_model", lambda _: FakeEmbeddings())

    assert ingest.ingest(config).chunks_stored == 2
    assert ingest.ingest(config).chunks_stored == 2
    assert ingest.ingest(config).chunks_stored == 2

    store = Chroma(
        collection_name="test",
        persist_directory=str(config.vector_store.directory),
        embedding_function=FakeEmbeddings(),
    )
    records = store.get(include=["documents", "metadatas"])
    assert sorted(records["documents"]) == ["bravo", "new alpha"]
    assert len(records["ids"]) == 2


def test_reset_removes_only_configured_collection(tmp_path, monkeypatch):
    import ingest

    config = app_config(tmp_path, collection="target")
    other = Chroma(
        collection_name="other",
        persist_directory=str(config.vector_store.directory),
        embedding_function=FakeEmbeddings(),
    )
    other.add_texts(["keep me"], ids=["other-id"])
    monkeypatch.setattr(
        ingest,
        "load_documents",
        lambda _: LoadReport([Document(page_content="new", metadata={"source": "a.txt"})]),
    )
    monkeypatch.setattr(ingest, "create_embedding_model", lambda _: FakeEmbeddings())

    ingest.ingest(config, reset=True)

    assert other.get()["ids"] == ["other-id"]
    target = Chroma(
        collection_name="target",
        persist_directory=str(config.vector_store.directory),
        embedding_function=FakeEmbeddings(),
    )
    assert len(target.get()["ids"]) == 1


def test_ingest_rejects_completely_empty_load(tmp_path, monkeypatch):
    import ingest

    monkeypatch.setattr(ingest, "load_documents", lambda _: LoadReport([]))

    with pytest.raises(ValueError, match="No documents"):
        ingest.ingest(app_config(tmp_path))


def test_main_parses_arguments_and_prints_summary(monkeypatch, capsys, tmp_path):
    import ingest

    config = app_config(tmp_path)
    calls = {}
    monkeypatch.setattr(ingest, "load_dotenv", lambda: calls.setdefault("dotenv", True))
    monkeypatch.setattr(ingest, "load_config", lambda path: calls.setdefault("path", path) and config)

    def fake_ingest(received, reset=False):
        calls["config"] = received
        calls["reset"] = reset
        return ingest.IngestReport(documents_loaded=3, chunks_stored=7, skipped=2, failures=1)

    monkeypatch.setattr(ingest, "ingest", fake_ingest)

    result = ingest.main(["--config", "custom.yaml", "--reset"])

    assert result == 0
    assert calls == {
        "dotenv": True,
        "path": "custom.yaml",
        "config": config,
        "reset": True,
    }
    output = capsys.readouterr().out
    assert "Documents loaded: 3" in output
    assert "Chunks stored: 7" in output
    assert "Unsupported files skipped: 2" in output
    assert "Load failures: 1" in output
