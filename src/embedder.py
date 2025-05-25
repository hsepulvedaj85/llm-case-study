from sentence_transformers import SentenceTransformer

class Embedder:
    """
    A class for generating text embeddings using a pre-trained SentenceTransformer model 
    snowflake-arctic-embed-s

    This class initializes a SentenceTransformer model and provides a method to encode a list of text 
    strings into numerical vector embeddings.

    Attributes:
        model (SentenceTransformer): The loaded SentenceTransformer model.
    
    """
    def __init__(self, model_name="Snowflake/snowflake-arctic-embed-s"):
        """
        Initializes the Embedder with a specified SentenceTransformer model.

        Args:
            model_name (str, optional): The name or path of the pre-trained SentenceTransformer 
                                        model to load.
                                        Defaults to "snowflake-arctic-embed-s".
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of text strings.

        Args:
            texts (list[str]): A list of strings to be embedded.

        Returns:
            list[list[float]]: A list of embeddings, where each embedding is a list of floats 
                                representing the numerical vector of the corresponding text. 
                                The progress of the embedding process is shown in the console.

        Example:
            >>> embedder = Embedder()
            >>> sentences = ["Hello world.", "How are you?"]
            >>> embeddings = embedder.embed_texts(sentences)
            >>> print(len(embeddings))
            2
            >>> print(len(embeddings[0])) # Embedding dimension
            768
        """
        return self.model.encode(texts, show_progress_bar=True)