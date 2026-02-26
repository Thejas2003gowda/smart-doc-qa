import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Smart Doc QA", page_icon="📄", layout="wide")
st.title("📄 Smart Doc QA")
st.caption("Upload PDFs and ask questions — answers come with source citations.")

# Sidebar: Upload
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            )
            if response.status_code == 200:
                data = response.json()
                st.success(
                    f"Indexed {data['chunks_created']} chunks "
                    f"from {data['pages_extracted']} pages"
                )
            else:
                st.error(f"Error: {response.text}")

    # Show index status
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.metric("Documents Indexed", health["documents_indexed"])
    except Exception:
        st.warning("API not running. Start with: uvicorn src.api.main:app --port 8000")

# Main: Chat
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Searching and generating answer..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": 5}
        )
        if response.status_code == 200:
            data = response.json()

            st.markdown("### Answer")
            st.markdown(data["answer"])

            with st.expander("📚 Sources"):
                for i, src in enumerate(data["sources"]):
                    st.markdown(
                        f"**Source {i+1}:** {src['source']}, "
                        f"Page {src['page']} "
                        f"(relevance: {src['score']:.1%})"
                    )
                    st.text(src["excerpt"])
                    st.divider()
        else:
            st.error(f"Error: {response.text}")