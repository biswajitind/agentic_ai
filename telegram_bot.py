import logging
import os

from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from rag_agent import RAGAgent


LOGGER = logging.getLogger(__name__)


def split_message(text: str, limit: int = 4096) -> list[str]:
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = max(text.rfind("\n", 0, limit + 1), text.rfind(" ", 0, limit + 1))
        if cut < 1:
            cut = limit
        else:
            cut += 1
        parts.append(text[:cut])
        text = text[cut:]
    return parts


async def start(update, context) -> None:
    await update.effective_message.reply_text("Send me a question about the indexed documents.")


def _source_lines(sources: list[dict]) -> list[str]:
    lines = []
    seen = set()
    for source in sources:
        key = (source["source"], source.get("page"))
        if key in seen:
            continue
        seen.add(key)
        label = source["source"]
        if source.get("page") is not None:
            label += f" (page {source['page']})"
        lines.append(label)
    return lines


async def answer(update, context) -> None:
    try:
        result = context.application.bot_data["agent"].ask(update.effective_message.text)
        source_lines = _source_lines(result["sources"])
        reply = result["answer"]
        if source_lines:
            reply += "\n\nSources:\n" + "\n".join(f"- {line}" for line in source_lines)
        for part in split_message(reply):
            await update.effective_message.reply_text(part)
    except Exception:
        LOGGER.exception("Failed to answer Telegram message")
        await update.effective_message.reply_text("I could not answer that question. Please try again.")


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")

    application = Application.builder().token(token).build()
    application.bot_data["agent"] = RAGAgent.from_config(
        os.getenv("PASHA_CONFIG", "config.yaml")
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))
    application.run_polling()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
