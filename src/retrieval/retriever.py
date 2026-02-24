from src.vectorstore.chroma_store import VectorStore


class Retriever:
    """Retrieve relevant documents with optional filtering."""

    def __init__(self, vector_store: VectorStore = None):
        self.store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.store.query(query, top_k=top_k)
        filtered = [r for r in results if r["similarity_score"] > 0.3]
        return filtered if filtered else results[:2]