from ollama import chat as ollama_chat

def generate_answer(query, context):
    system_prompt = (
        "You are a concise and direct assistant. Use ONLY the information provided in context. "
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer briefly, **use the question within to rephrasing the answer**.:"
    
    response = ollama_chat(
        model="llama3.2:1b",  # Must match the model tag you downloaded in Ollama
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={
            "temperature":0.1,
            "top_p": 0.1
        }
    )
    
    return response["message"]["content"]