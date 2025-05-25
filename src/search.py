from src.milvus_client import connect_milvus
from pymilvus import Collection

# Connect to Milvus and load collection
print(f"Connecting to Milvus...")
connect_milvus()
print(f"Connected to Milvus!")
print(f"Loading Collection...")
collection = Collection("voss_diary")
collection.load()
print(f"Collection loaded!")

def search_embedding(query_embedding, limit: int = 5):
    """
    Searches the Milvus collection for the most similar document chunks based on a given query 
    embedding.

    This function performs a similarity search within the 'voss_diary' collection using a COSINE 
    metric.
    It retrieves the top 'limit' number of chunks, extracts their text content and source, and 
    concatenates them into a single context string.

    Args:
        query_embedding (list[float]): The embedding vector of the search query.
        limit (int, optional): The maximum number of similar chunks to retrieve.
                                Defaults to 5.

    Returns:
        str: A concatenated string of the text content from the top similar
                chunks, separated by double newlines.

    Example:
        >>> # Assuming 'query_embedding' is a pre-computed embedding vector
        >>> # e.g., query_embedding = embedder.embed_texts(["What is Milvus?"])[0]
        >>> context = search_embedding(query_embedding, limit=3)
        >>> print(context)
        # (Output would be the text content of the 3 most similar chunks)
    """
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 1024}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["text", "source"]
    )
    # 4. Retrieve Top Chunks as Context
    top_chunks = [result.entity.get("text") for result in results[0]]
    context = "\n\n".join(top_chunks)
    return context