from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from sources import (
    AppConfig,
    ChunkingConfig,
    ModelConfig,
    RetrievalConfig,
    SourceConfig,
    VectorStoreConfig,
)


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        return self.documents


class FakeChatModel:
    def __init__(self, content="grounded answer"):
        self.content = content
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=self.content)


def app_config(tmp_path: Path, top_k: int = 4) -> AppConfig:
    return AppConfig(
        source=SourceConfig(str(tmp_path / "docs"), False, 2),
        vector_store=VectorStoreConfig(tmp_path / "chroma", "test"),
        chunking=ChunkingConfig(100, 10),
        retrieval=RetrievalConfig(top_k),
        models=ModelConfig("openai", "chat", "embed"),
    )


def test_ask_rejects_blank_question():
    from rag_agent import RAGAgent

    with pytest.raises(ValueError, match="question"):
        RAGAgent(FakeRetriever([]), FakeChatModel()).ask("   ")


def test_ask_builds_grounded_prompt_and_deduplicates_sources():
    from rag_agent import RAGAgent

    retriever = FakeRetriever(
        [
            Document(page_content="first context", metadata={"source": "guide.pdf", "page": 1}),
            Document(page_content="second context", metadata={"source": "guide.pdf", "page": 1}),
            Document(page_content="web context", metadata={"source": "https://example.com"}),
        ]
    )
    model = FakeChatModel()

    result = RAGAgent(retriever, model).ask("What is Pasha?")

    assert retriever.questions == ["What is Pasha?"]
    rendered = "\n".join(message.content for message in model.messages[0])
    assert "only" in rendered.lower()
    assert "first context" in rendered
    assert "second context" in rendered
    assert "guide.pdf" in rendered
    assert result == {
        "answer": "grounded answer",
        "sources": [
            {"source": "guide.pdf", "page": 1},
            {"source": "https://example.com", "page": None},
        ],
    }


def test_ask_with_no_context_does_not_call_model():
    from rag_agent import RAGAgent

    model = FakeChatModel()

    result = RAGAgent(FakeRetriever([]), model).ask("Unknown?")

    assert result == {
        "answer": "I don't know based on the available documents.",
        "sources": [],
    }
    assert model.messages == []


def test_ask_normalizes_non_string_content():
    from rag_agent import RAGAgent

    document = Document(page_content="context", metadata={"source": "notes.txt"})

    result = RAGAgent(FakeRetriever([document]), FakeChatModel(42)).ask("Question")

    assert result["answer"] == "42"


def test_from_config_forwards_top_k_and_builds_dependencies(tmp_path, monkeypatch):
    import rag_agent

    config = app_config(tmp_path, top_k=7)
    captured = {}
    retriever = FakeRetriever([])

    class FakeCollection:
        def count(self):
            return 1

    class FakeChroma:
        _collection = FakeCollection()

        def __init__(self, **kwargs):
            captured["store"] = kwargs

        def as_retriever(self, **kwargs):
            captured["retriever"] = kwargs
            return retriever

    monkeypatch.setattr(rag_agent, "load_config", lambda path: config)
    monkeypatch.setattr(rag_agent, "create_embedding_model", lambda _: "embeddings")
    monkeypatch.setattr(rag_agent, "create_chat_model", lambda _: "chat")
    monkeypatch.setattr(rag_agent, "Chroma", FakeChroma)

    agent = rag_agent.RAGAgent.from_config("custom.yaml")

    assert agent.retriever is retriever
    assert agent.chat_model == "chat"
    assert captured["store"] == {
        "collection_name": "test",
        "persist_directory": str(config.vector_store.directory),
        "embedding_function": "embeddings",
    }
    assert captured["retriever"] == {"search_type": "similarity", "search_kwargs": {"k": 7}}


def test_from_config_rejects_existing_empty_collection(tmp_path, monkeypatch):
    import rag_agent

    class EmptyCollection:
        def count(self):
            return 0

    class EmptyChroma:
        _collection = EmptyCollection()

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(rag_agent, "load_config", lambda path: app_config(tmp_path))
    monkeypatch.setattr(rag_agent, "create_embedding_model", lambda _: "embeddings")
    monkeypatch.setattr(rag_agent, "Chroma", EmptyChroma)

    with pytest.raises(ValueError, match="run ingestion"):
        rag_agent.RAGAgent.from_config("config.yaml")
