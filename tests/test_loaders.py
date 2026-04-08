import csv
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_engine.loaders import load_documents
from rag_engine.loaders.csv_loader import load_csv
from rag_engine.loaders.markdown_loader import load_markdown
from rag_engine.loaders.pdf_loader import load_pdf
from rag_engine.loaders.web_loader import load_web


class TestCSVLoader:
    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("term,definition\nRAG,Retrieval-Augmented Generation\nLLM,Large Language Model\n")

        docs = load_csv(csv_file)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata["file_type"] == "csv" for d in docs)

    def test_load_csv_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")

    def test_load_csv_wrong_extension(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .csv file"):
            load_csv(txt_file)


class TestMarkdownLoader:
    def test_load_markdown(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nSome content about RAG.\n\n## Section\n\nMore content.")

        docs = load_markdown(md_file)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata["file_type"] == "markdown" for d in docs)

    def test_load_markdown_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_markdown("nonexistent.md")


class TestPDFLoader:
    def test_load_pdf_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_pdf("nonexistent.pdf")

    def test_load_pdf_wrong_extension(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            load_pdf(txt_file)


class TestWebLoader:
    def test_web_loader_invalid_url(self):
        with pytest.raises(ValueError, match="Invalid URL"):
            load_web("not-a-url")


class TestLoadDocuments:
    def test_auto_detect_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\nval1,val2\n")

        docs = load_documents(str(csv_file))
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "csv"

    def test_auto_detect_markdown(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nWorld")

        docs = load_documents(str(md_file))
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "markdown"

    def test_explicit_source_type(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\nval1,val2\n")

        docs = load_documents(str(csv_file), source_type="csv")
        assert len(docs) >= 1

    def test_unknown_extension(self, tmp_path):
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text("hello")
        with pytest.raises(ValueError, match="Cannot infer source type"):
            load_documents(str(xyz_file))

    def test_unknown_source_type(self):
        with pytest.raises(ValueError, match="Unknown source_type"):
            load_documents("test.txt", source_type="unknown")
