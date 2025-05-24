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

def search_embedding(query_embedding):
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 128}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["text", "source"]
    )
    # 4. Retrieve Top Chunks as Context
    top_chunks = [result.entity.get("text") for result in results[0]]
    context = "\n\n".join(top_chunks)
    return context