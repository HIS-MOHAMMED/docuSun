import os

def discover_files(path):
    """
    Discover files from data directory

    Parameters:
    - None

    Returns:
    - A string representing a single file path or a list of strings representing multiple file paths.

    """
    #get all files into data directory
    files = os.listdir(path)
    #filter only pdf files
    pdf_files = [f'{path}/{file}' for file in files if file.endswith('e.pdf')]
    return pdf_files