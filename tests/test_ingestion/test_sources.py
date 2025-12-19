import os
import shutil
import pymupdf
from engine.ingestion.sources import discover_files

def create_temp_pdf(path):
    # Create a very small empty pdf with multiple pages
    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_text((72,72),"")

    page2 = doc.new_page()
    page2.insert_text((72,72), "")

    doc.save(path)
    doc.close()

def test_discover_files(tmp_path):
    # Arrange : create temporary data directory and files
    data_dir  = tmp_path/'tests'
    os.mkdir(data_dir)

    create_temp_pdf(data_dir/"sample.pdf")
    
    # Act: run function the function
    result = discover_files(data_dir)

    # Assert : only PDF file are detected
    assert len(result) == 1
    assert result[0].endswith('tests/sample.pdf')