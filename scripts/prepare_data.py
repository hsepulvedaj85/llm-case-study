"""
Extracts text content from a PDF and preprocesses it.

This function loads a PDF document, cleans its text content by removing
excessive whitespace, page numbers, and control characters, and then

Args:
    file_path (str): The path to the PDF file to be processed.
    
    Returns:
        list[Document]: A list of `langchain.schema.Document` objects, of the input PDF. Each `Document` includes `page_content`.
        
    Raises:
        FileNotFoundError: If the provided `file_path` does not exist.
        Exception: Catches and re-raises any exceptions encountered during PDF loading or processing.
    
    Example:
        >>> from langchain.schema import Document
        >>> # Assuming 'example.pdf' exists in the same directory
        >>> processed_docs = loadc_pdf("example.pdf")
"""
## Extracts text content, preprocess and splits into chunks

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document # Import Document class for cleaner metadata handling
import re # Import regular expressions for text cleaning


def loadc_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print('File loaded...')
    print('Pre-processing...')
    processed_docs = []
    for doc in docs:
        content = doc.page_content
        # 1. Eliminate multiple line breaks and excessive white space.
        # This collapses 2+ line breaks to 2, and spaces to 1.
        content = re.sub(r'\n{2,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        # 2. Remove common page breaks or headers/footers.
        # Single numbers at the end of the line
        content = re.sub(r'\s*Page \d+ of \d+\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s*\d+\s*$', '', content, flags=re.MULTILINE)
        # 3. Remove unwanted or control characters (if present)
        # This is more aggressive, use with caution.
        content = re.sub(r'[^\x00-\x7F]+', '', content)
        # 4. Final strip to eliminate blank spaces at the beginning/end of the document.
        content = content.strip()
        # If the document is not empty after preprocessing, add it
        if content:
            # Creates a new Document object with the cleaned content
            # Keeps the original metadata of the page.
            processed_docs.append(Document(page_content=content, metadata=doc.metadata))                    
    
    if not processed_docs:
        print("Warning: All documents were left empty after preprocessing.")
        return []
    
    print('Pre-processing completed...')
    
    return processed_docs