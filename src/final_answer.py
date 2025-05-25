"""
Generates a concise and direct answer to a query based solely on the provided context using the 
Ollama 'llama3.2:1b' model.

The function constructs a system prompt to guide the LLM towards conciseness and strict adherence 
to the given context. It then formulates a user prompt that includes the context and the question, 
instructing the model to rephrase the answer using the question itself.

Args:
    query (str): The question to be answered.
    context (str): The information to be used as the sole source for the answer. Search result.

    Returns:
        str: The LLM-model answer, rephrased with the question, based on the provided context.

    Example:
        >>> context_text = "The capital of France is Paris. Paris is known for the Eiffel Tower."
        >>> question = "What is the capital of France?"
        >>> answer = generate_answer(question, context_text)
        >>> print(answer)
        The capital of France is Paris.
"""
import sys    
from ollama import chat as ollama_chat

# Make sure that sys encoding is in UTF-8.
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    
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
            "temperature":0.1,  #Controls the "creativity" or randomness of the response.
            "top_p": 0.1        #Controls the diversity of the response.
        }
    )
    
    return response["message"]["content"]