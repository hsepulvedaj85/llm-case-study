"""
Extracts text content from a PDF, preprocesses it, and splits it into smaller, contextual chunks.

This function loads a PDF document, cleans its text content by removing
excessive whitespace, page numbers, and control characters, and then
divides the cleaned text into chunks of a specified size with a defined
overlap. Each chunk is enriched with metadata including its source
file path and a unique chunk ID.

Args:
    file_path (str): The path to the PDF file to be processed.
    chunk_size (int, optional): The maximum size of each text chunk. Defaults to 250 characters.
    chunk_overlap (int, optional): The number of characters to overlap between consecutive chunks.
    This helps maintain context across chunks. Defaults to 100 characters.

    Returns:
        list[Document]: A list of `langchain.schema.Document` objects, where each document represents 
        a processed and chunked portion of the input PDF. Each `Document` includes `page_content` 
        (the text chunk) and `metadata` (original page metadata, `chunk_id`, and `source`).
        
    Raises:
        FileNotFoundError: If the provided `file_path` does not exist.
        Exception: Catches and re-raises any exceptions encountered during PDF loading or processing.
    
    Example:
        >>> from langchain.schema import Document
        >>> # Assuming 'example.pdf' exists in the same directory
        >>> chunks = chunk_pdf("example.pdf", chunk_size=500, chunk_overlap=150)
        >>> for chunk in chunks:
        >>>     print(f"Content: {chunk.page_content[:100]}...")
        >>>     print(f"Metadata: {chunk.metadata}")
"""
## Extracts text content, preprocess and splits into chunks

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document # Import Document class for cleaner metadata handling
import re # Import regular expressions for text cleaning


def chunk_pdf(file_path: str, chunk_size: int = 250, chunk_overlap: int = 100):
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
    
    print('Pre-processing complete...')
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", " ", ""]
        ) ##This class allows to divide the text in an intelligent way, trying to keep the context and the structure.
    chunks = splitter.split_documents(processed_docs)
    
    # Add source for metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = file_path
        
    return chunks