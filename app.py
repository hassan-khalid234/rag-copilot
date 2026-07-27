import time
from dotenv import load_dotenv
from src.ingester import extract_text_from_pdf
from src.chunker import chunk_text_sliding_window
from src.vector_store import VectorStoreManager
from src.hybrid_retriever import HybridRAGRetriever
from src.generator import RAGGenerator
from src.logger import RAGLogger

load_dotenv()

def main():
    pdf_path = "data/sample.pdf"
    
    RAGLogger.log_step("Ingestion", "Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    RAGLogger.log_step("Chunking", "Splitting text into overlapping windows...")
    chunks = chunk_text_sliding_window(text, chunk_size=500, overlap=50)
    
    RAGLogger.log_step("Indexing", "Populating ChromaDB & BM25 sparse index...")
    vector_db = VectorStoreManager()
    hybrid_engine = HybridRAGRetriever(vector_db)
    hybrid_engine.index_documents(chunks)
    
    generator = RAGGenerator()
    query = "What is the primary topic of this document?"
    
    # Measure Retrieval Latency
    start_time = time.time()
    reranked_context = hybrid_engine.retrieve(query, fetch_k=10, final_top_k=3)
    duration_ms = (time.time() - start_time) * 1000
    
    # Log Observability Traces
    RAGLogger.log_retrieval(
        query=query, 
        raw_hits=10, 
        reranked_chunks=reranked_context, 
        duration_ms=duration_ms
    )
    
    # Generate Answer
    RAGLogger.log_step("Generation", "Sending prompt to Groq (llama-3.1-8b-instant)...")
    answer = generator.generate_answer(query, reranked_context)
    
    print(f"\n💡 GENERATED ANSWER:\n{answer}\n")

if __name__ == "__main__":
    main()