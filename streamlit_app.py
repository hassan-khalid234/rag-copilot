import streamlit as st
import time
from dotenv import load_dotenv

from src.ingester import extract_text_from_pdf
from src.chunker import chunk_text_sliding_window
from src.vector_store import VectorStoreManager
from src.hybrid_retriever import HybridRAGRetriever
from src.generator import RAGGenerator

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Production RAG Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Production RAG Copilot")
st.caption("Hybrid Search (BM25 + Dense) | Cross-Encoder Reranking | Llama 3.1 8B via Groq")

# Initialize persistent session state for vector DB and retriever
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Sidebar for PDF Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    chunk_size = st.slider("Chunk Size (chars)", 200, 1000, 500, step=50)
    overlap = st.slider("Chunk Overlap (chars)", 0, 200, 50, step=10)
    
    if uploaded_file and not st.session_state.indexed:
        with st.spinner("Extracting & Indexing document..."):
            # Save uploaded file temporarily
            temp_path = f"data/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run ingestion pipeline
            text = extract_text_from_pdf(temp_path)
            chunks = chunk_text_sliding_window(text, chunk_size=chunk_size, overlap=overlap)
            
            # Initialize Hybrid Vector Store
            vector_db = VectorStoreManager()
            hybrid_engine = HybridRAGRetriever(vector_db)
            hybrid_engine.index_documents(chunks)
            
            st.session_state.retriever = hybrid_engine
            st.session_state.indexed = True
            st.success(f"Indexed {len(chunks)} chunks successfully!")

# Main QA Interface
if st.session_state.indexed:
    user_query = st.text_input("Ask a question about your document:")
    
    if user_query:
        col1, col2 = st.columns([3, 2])
        
        with st.spinner("Retrieving & Reranking chunks..."):
            start_time = time.time()
            reranked_chunks = st.session_state.retriever.retrieve(user_query, fetch_k=10, final_top_k=3)
            retrieval_ms = (time.time() - start_time) * 1000
            
            generator = RAGGenerator()
            answer = generator.generate_answer(user_query, reranked_chunks)
        
        # Column 1: Generated Response
        with col1:
            st.subheader("💡 Answer")
            st.write(answer)
            st.caption(f"⚡ Retrieval latency: {retrieval_ms:.2f} ms")
            
        # Column 2: Source Citation Highlighting
        with col2:
            st.subheader("📌 Citation Sources")
            for i, chunk in enumerate(reranked_chunks, 1):
                with st.expander(f"Source Chunk #{i}", expanded=(i == 1)):
                    st.markdown(f"```text\n{chunk}\n```")
else:
    st.info("👈 Upload a PDF document in the sidebar to begin querying.")