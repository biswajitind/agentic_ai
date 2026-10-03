import os

import streamlit as st
from dotenv import load_dotenv

from rag_agent import RAGAgent


@st.cache_resource
def get_agent(config_path: str) -> RAGAgent:
    return RAGAgent.from_config(config_path)


def render() -> None:
    load_dotenv()
    st.title("Pasha RAG")
    agent = get_agent(os.getenv("PASHA_CONFIG", "config.yaml"))
    messages = st.session_state.setdefault("messages", [])

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        page = source.get("page")
                        label = source["source"]
                        st.write(f"{label} (page {page})" if page is not None else label)

    if question := st.chat_input("Ask a question about your documents"):
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        result = agent.ask(question)
        assistant = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
        messages.append(assistant)
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for source in result["sources"]:
                        page = source.get("page")
                        label = source["source"]
                        st.write(f"{label} (page {page})" if page is not None else label)


if __name__ == "__main__":
    render()
