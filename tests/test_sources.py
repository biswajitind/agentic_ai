from pathlib import Path

import pytest
from pypdf import PdfWriter

from sources import (
    AppConfig,
    ChunkingConfig,
    ModelConfig,
    RetrievalConfig,
    SourceConfig,
    VectorStoreConfig,
    load_documents,
    normalize_url,
)


def app_config(location: str, recursive: bool = False, max_depth: int = 2) -> AppConfig:
    return AppConfig(
        source=SourceConfig(location, recursive, max_depth),
        vector_store=VectorStoreConfig(Path("vectors"), "test"),
        chunking=ChunkingConfig(100, 10),
        retrieval=RetrievalConfig(4),
        models=ModelConfig("openai", "chat", "embedding"),
    )


def test_loads_one_text_file_with_source_metadata(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Pasha notes", encoding="utf-8")

    report = load_documents(app_config(str(source)))

    assert [document.page_content for document in report.documents] == ["Pasha notes"]
    assert report.documents[0].metadata["source"] == str(source.resolve())
    assert report.skipped == 0
    assert report.failures == []


def test_explicit_unsupported_file_is_an_error(tmp_path):
    source = tmp_path / "notes.csv"
    source.write_text("value", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file"):
        load_documents(app_config(str(source)))


def test_directory_recursion_and_skipped_file_count(tmp_path):
    (tmp_path / "top.md").write_text("top", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("skip", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.txt").write_text("deep", encoding="utf-8")

    flat = load_documents(app_config(str(tmp_path), recursive=False))
    recursive = load_documents(app_config(str(tmp_path), recursive=True))

    assert [doc.page_content for doc in flat.documents] == ["top"]
    assert flat.skipped == 1
    assert sorted(doc.page_content for doc in recursive.documents) == ["deep", "top"]
    assert recursive.skipped == 1


def test_pdf_retains_page_number_metadata(tmp_path):
    source = tmp_path / "one-page.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with source.open("wb") as output:
        writer.write(output)

    report = load_documents(app_config(str(source)))

    assert len(report.documents) == 1
    assert report.documents[0].metadata["source"] == str(source.resolve())
    assert report.documents[0].metadata["page"] == 0


def test_normalize_url_removes_fragments_and_equivalent_trailing_slash():
    assert normalize_url("https://Example.com/docs/#part") == "https://example.com/docs"
    assert normalize_url("https://example.com/docs") == "https://example.com/docs"


def test_web_crawl_is_same_host_cycle_safe_and_depth_limited():
    pages = {
        "https://example.com": '<main>home</main><a href="/a#one">A</a><a href="https://other.com/x">X</a>',
        "https://example.com/a": '<main>alpha</main><a href="/">Home</a><a href="/deep">Deep</a>',
        "https://example.com/deep": "<main>deep</main>",
    }
    calls = []

    def fetch(url):
        calls.append(url)
        return pages[url]

    shallow = load_documents(app_config("https://example.com/", True, 1), fetch)

    assert [doc.metadata["source"] for doc in shallow.documents] == [
        "https://example.com",
        "https://example.com/a",
    ]
    assert calls == ["https://example.com", "https://example.com/a"]


def test_web_page_without_recursion_loads_only_start_page():
    html = '<main>home</main><a href="/child">Child</a>'

    report = load_documents(app_config("https://example.com", False), lambda _: html)

    assert len(report.documents) == 1
    assert report.documents[0].metadata["source"] == "https://example.com"


def test_failed_child_page_keeps_successful_pages_and_reports_failure():
    pages = {
        "https://example.com": '<main>home</main><a href="/good">Good</a><a href="/bad">Bad</a>',
        "https://example.com/good": "<main>good</main>",
    }

    def fetch(url):
        if url.endswith("/bad"):
            raise OSError("connection lost")
        return pages[url]

    report = load_documents(app_config("https://example.com", True, 1), fetch)

    assert [doc.metadata["source"] for doc in report.documents] == [
        "https://example.com",
        "https://example.com/good",
    ]
    assert len(report.failures) == 1
    assert "https://example.com/bad" in report.failures[0]
