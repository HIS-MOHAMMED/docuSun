# discouver files and folders from data directory
import os

def discover_files():
    directory = 'data'
    #get all files into data directory
    files = os.listdir(directory)
    #filter only pdf files
    pdf_files = [f'{directory}/{file}' for file in files if file.endswith('.pdf')]
    return pdf_files