from dotenv import load_dotenv
from src.ingester import extract_text_from_pdf
from src.chunker import chunk_text_sliding_window
from src.vector_store import VectorStoreManager
from src.generator import RAGGenerator

load_dotenv()

def main():
    pdf_path = "data/sample.pdf"
    
    print("1. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    print("2. Chunking text...")
    chunks = chunk_text_sliding_window(text, chunk_size=500, overlap=50)
    
    print("3. Indexing into ChromaDB...")
    vector_db = VectorStoreManager()
    vector_db.add_documents(chunks)
    
    # Simple query loop
    generator = RAGGenerator()
    query = "What is the primary topic of this document?"
    
    print(f"\n🔍 Query: {query}")
    context = vector_db.query_similarity(query, top_k=3)
    answer = generator.generate_answer(query, context)
    
    print(f"\n💡 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()