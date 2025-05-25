from src.milvus_client import connect_milvus
from pymilvus import Collection
from sentence_transformers import CrossEncoder

print("Cargando modelo de re-ranking...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512)
print("Modelo de re-ranking cargado.")
# Connect to Milvus and load collection
print(f"Connecting to Milvus...")
connect_milvus()
print(f"Connected to Milvus!")
print(f"Loading Collection...")
collection = Collection("voss_diary")
collection.load()
print(f"Collection loaded!")

def search_embedding(query_embedding: list[float], original_query: str):
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 1024}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["text", "source"]
    )
    
    milvus_retrieved_chunks = []
    for hit in results[0]: 
        milvus_retrieved_chunks.append({
            "text": hit.entity.get("text"),
            "chunk_id": hit.entity.get("chunk_id"),
            "source": hit.entity.get("source"),
            "score": hit.score # Guardar el score de Milvus también
        })
        
    if not milvus_retrieved_chunks:
        print("No se encontraron chunks en Milvus.")
        return ""
    # Paso 2: Preparar pares (query, chunk_text) para el re-ranker
    sentence_pairs = [[original_query, chunk["text"]] for chunk in milvus_retrieved_chunks]

    print("Re-rankeando los chunks...")
    # El modelo de re-ranking asigna un score de relevancia a cada par
    rerank_scores = reranker_model.predict(sentence_pairs)
    # Paso 3: Combinar los scores de re-ranking con los chunks originales
    for i, chunk in enumerate(milvus_retrieved_chunks):
        chunk['rerank_score'] = rerank_scores[i]
    # Ordenar los chunks por el score de re-ranking (descendente)
    reranked_chunks = sorted(milvus_retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)
    
    # Paso 4: Seleccionar los top_k finales después del re-ranking
    # Este es el número de chunks que finalmente se envían al LLM.
    # Recomendado: 3 a 5 para Llama 3.2-1B
    TOP_K_RERANKED = 5 
    
    final_top_chunks = reranked_chunks[:TOP_K_RERANKED]
    
    # 4. Retrieve Top Chunks as Context for the LLM
    context = "\n\n".join([chunk["text"] for chunk in final_top_chunks])
    
    print(f"Re-ranking completado. Se seleccionaron {len(final_top_chunks)} chunks finales.")
    # Opcional: Imprimir los top chunks re-rankeados para depuración
    # for i, chunk in enumerate(final_top_chunks):
    #     print(f"Chunk {i+1} (Score: {chunk['rerank_score']:.4f}): {chunk['text'][:100]}...")
        
    return context