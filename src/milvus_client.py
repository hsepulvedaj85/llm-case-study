
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import utility


def connect_milvus(host="localhost", port="19530"):
    connections.connect(alias="default", host=host, port=port)

def setup_collection(collection_name="voss_diary"):
    if utility.has_collection(collection_name):
        utility.drop_collection("voss_diary")
        #return Collection(collection_name)
    
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    ]

    schema = CollectionSchema(fields, description="Chunks from Dr. Voss's Diary")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})
    collection.load()
    
    return collection

def insert_chunks(collection, chunks, embeddings):
    data = [
        [chunk.metadata["chunk_id"] for chunk in chunks],
        embeddings,
        [chunk.page_content for chunk in chunks],
        [chunk.metadata["source"] for chunk in chunks],
    ]
    collection.insert(data)