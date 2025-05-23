# documentar funcion
## Extracts text content and splits into chunks

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_pdf(file_path: str, chunk_size: int = 250, chunk_overlap: int = 100):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print('File loaded...')
    ## Ingresar preprocesamiento
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,  # Chunking con Solapamiento (Overlap)
        separators = ["\n\n", "\n", " ", ""]
        ) ##Esta clase permite dividir el texto de forma inteligente, tratando de mantener el contexto y la estructura.
    chunks = splitter.split_documents(docs)
    
    # Add page number and source for metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = file_path
        
    return chunks