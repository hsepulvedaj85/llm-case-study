from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import Collection
from sentence_transformers import SentenceTransformer
from ollama import chat as ollama_chat
from src.milvus_client import connect_milvus

app = FastAPI(title="RAG API with LLaMA 3.2-1B and Milvus")

# Load models and connect to Milvus once
connect_milvus()
collection = Collection("voss_diary")
collection.load()

embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")

# Request schema
class QueryRequest(BaseModel):
    question: str
    
# Response schema
class QueryResponse(BaseModel):
    answer: str
    #top_chunks: list[str]

# Helper: Perform vector search
def search_milvus(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query])[0]
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "text", "source"]
    )
    return [hit.entity.get("text") for hit in results[0]]

# Helper: Send prompt to LLaMA via Ollama
def ask_llama(query: str, context_chunks: list[str]):
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant answering questions using the provided context. "
        "Only use information in the context. Be accurate, and clear."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = ollama_chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response["message"]["content"]

# API Endpoint
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        chunks = search_milvus(request.question)
        answer = ask_llama(request.question, chunks)
        return QueryResponse(answer=answer, top_chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))