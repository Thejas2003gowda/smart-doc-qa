from src.ingestion.chunking import chunk_documents


def test_chunk_creates_correct_number():
    docs = [{"content": "word " * 200, "metadata": {"source": "test.pdf", "page": 1}}]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunk_preserves_metadata():
    docs = [{"content": "Hello world. " * 50, "metadata": {"source": "a.pdf", "page": 3}}]
    chunks = chunk_documents(docs, chunk_size=100)
    for c in chunks:
        assert c["metadata"]["source"] == "a.pdf"
        assert c["metadata"]["page"] == 3
        assert "chunk_index" in c["metadata"]


def test_empty_input():
    chunks = chunk_documents([])
    assert chunks == []