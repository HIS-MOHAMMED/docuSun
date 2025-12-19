from unittest.mock import patch, MagicMock
from engine.ingestion.loaders import load_pdf


def test_load_pdf():
    # Arrange
    fake_pdf = MagicMock()
    fake_pdf.__len__.return_value = 2  # pretend the PDF has 2 pages

    # Fake page 1
    fake_page1 = MagicMock()
    fake_page1.get_text.return_value = "Welcome to docuSun"

    # Fake page 2
    fake_page2 = MagicMock()
    fake_page2.get_text.return_value = ", have a good time."

    # doc.load_page(i)
    fake_pdf.load_page.side_effect = [fake_page1, fake_page2]

    files = ["test.pdf"]

    # Patch pymupdf.open inside the loaders module
    with patch("engine.ingestion.loaders.pymupdf.open", return_value=fake_pdf):
        # Act
        result = load_pdf(files)

    # Assert
    assert len(result) == 1
    doc = result[0]

    assert doc.metadata["source"] == "test.pdf"
    assert doc.page_content == "Welcome to docuSun, have a good time."
