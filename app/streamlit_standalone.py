import streamlit as st
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import tempfile

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import chunk_documents
from src.vectorstore.chroma_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

st.set_page_config(page_title="Smart Doc QA", page_icon="📄", layout="wide")
st.title("📄 Smart Doc QA")
st.caption("Upload PDFs and ask questions — answers come with source citations.")

# Initialize once per session
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
    st.session_state.retriever = Retriever(st.session_state.store)
    st.session_state.generator = Generator()

store = st.session_state.store
retriever = st.session_state.retriever
generator = st.session_state.generator

# Sidebar: Upload
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.getvalue())
                tmp_path = f.name
            docs = load_pdf(tmp_path)
            chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
            store.add_documents(chunks)
            os.unlink(tmp_path)
            st.success(f"Indexed {len(chunks)} chunks from {len(docs)} pages")
    st.metric("Chunks Indexed", store.get_collection_count())

# Main: Chat
question = st.text_input("Ask a question about your documents:")
if question:
    if store.get_collection_count() == 0:
        st.warning("Upload a PDF first.")
    else:
        with st.spinner("Searching and generating answer..."):
            retrieved = retriever.retrieve(question, top_k=5)
            result = generator.generate(question, retrieved)

            st.markdown("### Answer")
            st.markdown(result["answer"])

            with st.expander("📚 Sources"):
                for i, src in enumerate(result["sources"]):
                    st.markdown(
                        f"**Source {i+1}:** {src['source']}, "
                        f"Page {src['page']} "
                        f"(relevance: {src['score']:.1%})"
                    )
                    st.text(src["excerpt"])
                    st.divider()