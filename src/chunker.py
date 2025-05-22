## Extracts text content and splits into chunks

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE_PATH = "./data/dr_voss_diary.pdf"

def chunk_pdf(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print('File loaded...')
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    
    # Add page number and source for metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = FILE_PATH
        
    return chunks