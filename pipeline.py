# pipeline

from src.chunker import chunk_pdf
from src.embedder import Embedder
from src.milvus_client import connect_milvus, setup_collection, insert_chunks

def run_pipeline(pdf_path):
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