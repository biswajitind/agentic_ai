import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sources import ModelConfig


def _validate(config: ModelConfig) -> None:
    if config.provider.lower() != "openai":
        raise ValueError(f"Unsupported model provider: {config.provider}")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")


def create_chat_model(config: ModelConfig) -> BaseChatModel:
    _validate(config)
    return ChatOpenAI(model=config.chat)


def create_embedding_model(config: ModelConfig) -> Embeddings:
    _validate(config)
    return OpenAIEmbeddings(model=config.embedding)
