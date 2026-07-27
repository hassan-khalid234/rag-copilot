from dotenv import load_dotenv
from src.ingester import extract_text_from_pdf
from src.chunker import chunk_text_sliding_window
from src.vector_store import VectorStoreManager
from src.hybrid_retriever import HybridRAGRetriever
from src.generator import RAGGenerator

load_dotenv()

def main():
    pdf_path = "data/sample.pdf"
    
    print("1. Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    
    print("2. Chunking text...")
    chunks = chunk_text_sliding_window(text, chunk_size=500, overlap=50)
    
    print("3. Indexing into Hybrid Engine (Dense + BM25)...")
    vector_db = VectorStoreManager()
    hybrid_engine = HybridRAGRetriever(vector_db)
    hybrid_engine.index_documents(chunks)
    
    generator = RAGGenerator()
    query = "What is the primary topic of this document?"
    
    print(f"\n🔍 Query: {query}")
    print("4. Executing Hybrid Retrieval + Reranking...")
    reranked_context = hybrid_engine.retrieve(query, fetch_k=10, final_top_k=3)
    
    print("5. Generating response via Groq...")
    answer = generator.generate_answer(query, reranked_context)
    
    print(f"\n💡 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()