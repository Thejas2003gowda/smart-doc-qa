from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, os

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import chunk_documents
from src.vectorstore.chroma_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

app = FastAPI(title="Smart Doc QA", version="1.0.0")
store = VectorStore()
retriever = Retriever(store)
generator = Generator()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class AnswerResponse(BaseModel):
    answer: str
    sources: list
    model: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "documents_indexed": store.get_collection_count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_pdf(file_path)
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    count = store.add_documents(chunks)

    return {
        "filename": file.filename,
        "pages_extracted": len(docs),
        "chunks_created": count,
        "total_indexed": store.get_collection_count()
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if store.get_collection_count() == 0:
        raise HTTPException(400, "No documents indexed. Upload a PDF first.")

    docs = retriever.retrieve(request.question, top_k=request.top_k)
    result = generator.generate(request.question, docs)
    return AnswerResponse(**result)