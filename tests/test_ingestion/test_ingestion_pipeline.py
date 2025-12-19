from engine.ingestion.ingestion_pipeline import ingest_documents
from langchain.docstore.document import Document
import fitz  # PyMuPDF
import os

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def create_temp_pdf(path):
    # Create a very small PDF with multiple pages.
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "DocuSun A local Retrieval Augmented Generation (RAG) engine that" \
    " lets you upload, index, and query your documents privately using open-source models.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Give it a try.")

    doc.save(path)
    doc.close()


def test_ingestion_pipeline(tmp_path):
    # Arrange: Create a temp directory and PDF inside it
    data_dir = tmp_path / "tests"
    data_dir.mkdir()

    create_temp_pdf(data_dir / "sample.pdf")

    # Act: run the function
    chunks = ingest_documents(
        200,
        EMBEDDING_MODEL_NAME,
        data_dir
    )

    # Assert
    assert len(chunks) == 1
    assert chunks[0].metadata["source"].endswith("tests/sample.pdf")
    assert isinstance(chunks[0], Document)
    assert "DocuSun A local Retrieval Augmented Generation" in chunks[0].page_content
