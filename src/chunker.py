"""
Splits text content from a extracted and cleaned PDF content, into smaller, contextual chunks.

This function divides the cleaned text into chunks of a specified size with a defined
overlap. Each chunk is enriched with metadata including its source
file path and a unique chunk ID.

Args:
    List_docs[list]:  
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
        >>> processed_pdf = loadc_pdf(pdf_path)    
        >>> chunks = chunk_pdf(preproccesed_docs,"example.pdf", chunk_size=500, chunk_overlap=150)
        >>> for chunk in chunks:
        >>>     print(f"Content: {chunk.page_content[:100]}...")
        >>>     print(f"Metadata: {chunk.metadata}")
"""
## Extracts text content, preprocess and splits into chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_pdf(List_docs, file_path: str, chunk_size: int = 250, chunk_overlap: int = 100):
    docs = List_docs
    print('List loaded...')
    print('Pre-processing...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", " ", ""]
        ) ##This class allows to divide the text in an intelligent way, trying to keep the context and the structure.
    chunks = splitter.split_documents(docs)
    
    # Add source for metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = file_path
        
    return chunks