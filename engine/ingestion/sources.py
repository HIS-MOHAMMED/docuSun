import os

def discover_files():
    """
    Discover files from data directory

    Parameters:
    - None

    Returns:
    - A string representing a single file path or a list of strings representing multiple file paths.

    """
    directory = 'data'
    #get all files into data directory
    files = os.listdir(directory)
    #filter only pdf files
    pdf_files = [f'{directory}/{file}' for file in files if file.endswith('.pdf')]
    return pdf_files