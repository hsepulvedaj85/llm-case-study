# pipeline

from src.chunker import chunk_pdf
from src.embedder import Embedder
from src.milvus_client import connect_milvus, setup_collection, insert_chunks

def run_pipeline(pdf_path):
    """
    Executes the entire RAG (Retrieval-Augmented Generation) ingestion pipeline for a given PDF 
    document as a context.
    
    This function performs the following steps:
    1. **Chunks the PDF:** Reads the PDF, extracts text, preprocesses it, and splits it into 
        manageable chunks.
    2. **Generates Embeddings:** Creates numerical vector representations (embeddings) for each 
        of the generated text chunks using a pre-trained SentenceTransformer model.
    3. **Stores in Milvus:** Connects to the Milvus vector database, sets up a dedicated collection 
        (or re-initializes it if it exists), and inserts the chunks along with their embeddings for 
        efficient retrieval.

    Args:
        pdf_path (str): The file path to the PDF document to be processed and ingested.

    Example:
        >>> run_pipeline("./data/dr_voss_diary.pdf")
        # This will process the PDF, generate embeddings, and store them in Milvus.
    """
    print("PDF Loading...")
    # Task #1: Chunk the PDF
    chunks = chunk_pdf(pdf_path)
    print('PDF was chunked...')
    # Step 2: Generate embeddings
    embedder = Embedder()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_texts(texts)
    print("Finish embeddings")
    print(f"Embedding shape: {len(embeddings)} × {len(embeddings[0])}")
    print(f"Ingested chunks: {len(chunks)}")
    print(f"Embeddings: {len(embeddings)}")
        
    # Step 3: Store in Milvus
    connect_milvus()
    print("connect_milvus()")
    collection = setup_collection()
    print("collection = setup_collection()")
    print(f"collection has {collection.num_entities}")
    insert_chunks(collection, chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks into Milvus.")

if __name__ == "__main__":
    run_pipeline("./data/dr_voss_diary.pdf")