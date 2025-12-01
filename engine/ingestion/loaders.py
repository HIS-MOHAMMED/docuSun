import pymupdf
from langchain.docstore.document import Document
from engine.ingestion.cleaners import clean_extra_whitespaces, group_broken_paragraphs

def load_pdf(files):
    """
    Loads documents from PDF files using PyMuPDF

    Parameters:
    - files : A String representing a single file path or a list of strings representing multiple file paths
    
    Returns:
    - A list of Documens objects loaded from the provided pdf files
    
    Raises:
    - FileNotFoundException: If any of provided file pahts do not exists.
    - Exception: For any other issues encountered during file loading
    """
    if not isinstance(files, list):
        files = [files]
    documents = []
    for file_path in files:
        try:
            # Open the PDF file
            doc = pymupdf.open(file_path)
            
            # Extract text from each page
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text")
            # Apply post-processing steps
            text = clean_extra_whitespaces(text)
            text = group_broken_paragraphs(text)

            # Create a Document object 
            document = Document(
                page_content=text,
                metadata={"source": file_path}
            )

            # Add the Document object to documents list
            documents.append(document)      

        except FileNotFoundError as exception:
            print(f"file not found: {exception.filename}")
            raise
        except Exception as exception:
            print(f"An error occurred while loading {file_path}: {exception}")
            raise 
    return documents       