import chromadb
from src.config import CHROMA_PERSIST_DIR
from src.embeddings.embed import EmbeddingService


class VectorStore:
    """ChromaDB-backed vector store with persistent storage."""

    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = EmbeddingService()

    def add_documents(self, chunks: list[dict]) -> int:
        """Add chunked documents to the vector store."""
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.embed_texts(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Retrieve most relevant chunks for a query."""
        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i]
            })
        return retrieved

    def get_collection_count(self) -> int:
        return self.collection.count()