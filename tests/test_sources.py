import os
from engine.ingestion.sources import discover_files

def test_discover_files(tmp_path):
    # Arrange : create temporary data directory and files
    data_dir  = tmp_path/'data'
    os.mkdir(data_dir)
    (data_dir / 'a.pdf').touch()
    (data_dir / 'b.txt').touch()
    (data_dir / 'c.pdf').touch()

    # Act: temporarily swich to that folder and run function
    cwd = os.getcwd()
    os.chdir(tmp_path)
    result = discover_files()
    os.chdir(cwd)

    # Assert : only PDF file are detected
    expected = [f'data/a.pdf',f'data/c.pdf']
    assert sorted(result) == sorted(expected)