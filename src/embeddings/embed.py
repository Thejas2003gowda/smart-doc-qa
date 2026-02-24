from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class EmbeddingService:
    """Generate embeddings using a local sentence-transformer (FREE, no API key)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query).tolist()