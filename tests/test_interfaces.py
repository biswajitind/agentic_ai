import asyncio
import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest


class FakeMessage:
    def __init__(self, text=""):
        self.text = text
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


def test_split_message_preserves_text_and_telegram_limit():
    from telegram_bot import split_message

    text = ("word " * 1000) + ("x" * 5000)
    parts = split_message(text)

    assert "".join(parts) == text
    assert all(0 < len(part) <= 4096 for part in parts)


def test_start_sends_usage_message():
    from telegram_bot import start

    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)

    asyncio.run(start(update, None))

    assert len(message.replies) == 1
    assert "question" in message.replies[0].lower()


def test_answer_calls_shared_agent_and_appends_unique_sources():
    from telegram_bot import answer

    class Agent:
        def __init__(self):
            self.questions = []

        def ask(self, question):
            self.questions.append(question)
            return {
                "answer": "Use the deployment guide.",
                "sources": [
                    {"source": "guide.pdf", "page": 2},
                    {"source": "guide.pdf", "page": 2},
                    {"source": "https://example.com", "page": None},
                ],
            }

    agent = Agent()
    message = FakeMessage("How do I deploy?")
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(application=SimpleNamespace(bot_data={"agent": agent}))

    asyncio.run(answer(update, context))

    assert agent.questions == ["How do I deploy?"]
    reply = "".join(message.replies)
    assert reply.count("guide.pdf (page 2)") == 1
    assert reply.count("https://example.com") == 1


def test_main_names_missing_telegram_token(monkeypatch):
    import telegram_bot

    monkeypatch.setattr(telegram_bot, "load_dotenv", lambda: None)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        telegram_bot.main()


def test_main_uses_pasha_config_override(monkeypatch):
    import telegram_bot

    captured = {}

    class FakeRAGAgent:
        @classmethod
        def from_config(cls, path):
            captured["config"] = path
            return "agent"

    class FakeApplication:
        def __init__(self):
            self.bot_data = {}
            self.handlers = []

        def add_handler(self, handler):
            self.handlers.append(handler)

        def run_polling(self):
            captured["polled"] = True

    application = FakeApplication()

    class Builder:
        def token(self, token):
            captured["token"] = token
            return self

        def build(self):
            return application

    monkeypatch.setattr(telegram_bot, "load_dotenv", lambda: None)
    monkeypatch.setattr(telegram_bot, "RAGAgent", FakeRAGAgent)
    monkeypatch.setattr(telegram_bot.Application, "builder", lambda: Builder())
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("PASHA_CONFIG", "/tmp/custom.yaml")

    telegram_bot.main()

    assert captured == {
        "config": "/tmp/custom.yaml",
        "token": "bot-token",
        "polled": True,
    }
    assert application.bot_data["agent"] == "agent"
    assert len(application.handlers) == 2


def test_importing_streamlit_module_has_no_runtime_side_effect(monkeypatch):
    sys.modules.pop("app", None)
    rag_module = importlib.import_module("rag_agent")
    monkeypatch.setattr(
        rag_module.RAGAgent,
        "from_config",
        classmethod(lambda cls, path: pytest.fail("agent created during import")),
    )

    imported = importlib.import_module("app")

    assert callable(imported.render)


def test_all_production_modules_import_without_starting_services():
    for module_name in ["sources", "models", "ingest", "rag_agent", "app", "telegram_bot"]:
        assert importlib.import_module(module_name)


def test_ingest_help_needs_no_credentials_and_lists_options():
    result = subprocess.run(
        [sys.executable, "ingest.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--reset" in result.stdout
