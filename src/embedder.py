## src/embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="Snowflake/snowflake-arctic-embed-s"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=True)
