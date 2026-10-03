from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from models import create_chat_model, create_embedding_model
from sources import load_config


class RAGAgent:
    def __init__(self, retriever, chat_model) -> None:
        self.retriever = retriever
        self.chat_model = chat_model

    @classmethod
    def from_config(cls, path: str | Path) -> "RAGAgent":
        config = load_config(path)
        embeddings = create_embedding_model(config.models)
        store = Chroma(
            collection_name=config.vector_store.collection,
            persist_directory=str(config.vector_store.directory),
            embedding_function=embeddings,
        )
        if store._collection.count() == 0:
            raise ValueError("The configured Chroma collection is empty; run ingestion first")
        retriever = store.as_retriever(
            search_type="similarity", search_kwargs={"k": config.retrieval.top_k}
        )
        return cls(retriever, create_chat_model(config.models))

    def ask(self, question: str) -> dict[str, object]:
        question = question.strip()
        if not question:
            raise ValueError("A non-empty question is required")

        documents = self.retriever.invoke(question)
        if not documents:
            return {
                "answer": "I don't know based on the available documents.",
                "sources": [],
            }

        context_parts = []
        sources = []
        seen = set()
        for document in documents:
            source = str(document.metadata.get("source", "unknown"))
            page = document.metadata.get("page")
            context_parts.append(f"Source: {source}; page: {page}\n{document.page_content}")
            key = (source, page)
            if key not in seen:
                seen.add(key)
                sources.append({"source": source, "page": page})

        context = "\n\n".join(context_parts)
        messages = [
            SystemMessage(
                content=(
                    "Answer only from the supplied context. If it does not contain the answer, "
                    "say that you don't know based on the available documents."
                )
            ),
            HumanMessage(content=f"Context:\n\n{context}\n\nQuestion: {question}"),
        ]
        response = self.chat_model.invoke(messages)
        return {"answer": str(response.content), "sources": sources}
