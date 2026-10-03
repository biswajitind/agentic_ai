import pytest

from sources import ModelConfig


def test_chat_factory_forwards_configured_model(monkeypatch):
    import models

    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return "chat-client"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(models, "ChatOpenAI", fake_chat)

    result = models.create_chat_model(ModelConfig("openai", "chosen-chat", "embed"))

    assert result == "chat-client"
    assert captured == {"model": "chosen-chat"}


def test_embedding_factory_forwards_configured_model(monkeypatch):
    import models

    captured = {}

    def fake_embeddings(**kwargs):
        captured.update(kwargs)
        return "embedding-client"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(models, "OpenAIEmbeddings", fake_embeddings)

    result = models.create_embedding_model(ModelConfig("openai", "chat", "chosen-embed"))

    assert result == "embedding-client"
    assert captured == {"model": "chosen-embed"}


@pytest.mark.parametrize("factory_name", ["create_chat_model", "create_embedding_model"])
def test_model_factory_names_missing_openai_key(monkeypatch, factory_name):
    import models

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        getattr(models, factory_name)(ModelConfig("openai", "chat", "embed"))


@pytest.mark.parametrize("factory_name", ["create_chat_model", "create_embedding_model"])
def test_model_factory_rejects_unsupported_provider(monkeypatch, factory_name):
    import models

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError, match="Unsupported model provider.*anthropic"):
        getattr(models, factory_name)(ModelConfig("anthropic", "chat", "embed"))
