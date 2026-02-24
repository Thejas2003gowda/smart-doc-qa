from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "recursive"
) -> list[dict]:
    """Split documents into chunks while preserving metadata."""

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for j, split in enumerate(splits):
            chunks.append({
                "content": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": j,
                    "chunk_strategy": strategy,
                    "chunk_size": chunk_size
                }
            })
    return chunks