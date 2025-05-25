
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import utility


def connect_milvus(host="localhost", port="19530"):
    """
    Establishes a connection to a Milvus vector database.

    Args:
        host (str, optional): The hostname or IP address of the Milvus instance.
                                Defaults to "localhost".
        port (str, optional): The port number of the Milvus instance.
                                Defaults to "19530".
    """
    connections.connect(alias="default", host=host, port=port)

def setup_collection(collection_name="voss_diary"):
    """
    Sets up a Milvus collection for storing document chunks and their embeddings.

    If a collection with the given name already exists, it will be dropped (primarily for 
    development purposes as indicated in the original code) and a new one will be created. 
    The collection schema includes fields for chunk ID, embedding vector (dimension 384), 
    text content, and source. An IVF_FLAT index with COSINE metric type is created on the 
    embedding field.

    Args:
        collection_name (str, optional): The name of the collection to set up.
                                            Defaults to "voss_diary".

    Returns:
        Collection: The newly created and loaded Milvus Collection object.
    """
    if utility.has_collection(collection_name):
        utility.drop_collection("voss_diary")  #usin to devlope
        #return Collection(collection_name)
    
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    ]

    schema = CollectionSchema(fields, description="Chunks from Dr. Voss's Diary")
    collection = Collection(name=collection_name, schema=schema)
    idex_params = {
        "index_type": "IVF_FLAT", 
        "metric_type": "COSINE", 
        "params": {"nlist": 1024}
    }
    collection.create_index("embedding", idex_params)
    collection.load()
    
    return collection

def insert_chunks(collection, chunks, embeddings):
    """
    Inserts processed document chunks and their corresponding embeddings into a Milvus collection.

    This function takes a list of `Document` objects (chunks) and their generated embeddings, then
    organizes this data for insertion into the specified Milvus collection. After insertion, 
    `collection.flush()` is called to ensure the data is written to disk.

    Args:
        collection (Collection): The Milvus Collection object where the data will be inserted.
        chunks (list): A list of `langchain.schema.Document` objects, each representing
                       a text chunk with `page_content` and `metadata`.
        embeddings (list[list[float]]): A list of embedding vectors, corresponding
                                        to each chunk in the `chunks` list.
    """
    data = [
        [chunk.metadata["chunk_id"] for chunk in chunks],
        embeddings,
        [chunk.page_content for chunk in chunks],
        [chunk.metadata["source"] for chunk in chunks],
    ]
    collection.insert(data)
    collection.flush() # Importante para asegurar que los datos se escriban a disco
    print("Chunks inserted and flushed!")