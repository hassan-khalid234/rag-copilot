import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize local embedding model & ChromaDB in memory
print("Loading local embedding model...")
default_ef = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="rag_baseline", 
    embedding_function=default_ef
)

# 2. Extract & Chunk text from PDF
def load_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += (chunk_size - overlap)
    return chunks

# 3. Add chunks to local ChromaDB
pdf_filename = "sample.pdf"
if os.path.exists(pdf_filename):
    chunks = load_and_chunk_pdf(pdf_filename)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")
else:
    print(f"Notice: '{pdf_filename}' not found. Place a sample.pdf in this directory.")

# 4. Query Engine (Powered by Groq)
def query_rag(user_query):
    # Retrieve top 3 relevant chunks
    results = collection.query(query_texts=[user_query], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        return "No documents indexed.", []

    retrieved_texts = results['documents'][0]
    context = "\n---\n".join(retrieved_texts)

    prompt = f"""
Answer the question based ONLY on the provided context below.

Context:
{context}

Question: {user_query}
"""
    # Reads GROQ_API_KEY from environment automatically
    client = Groq()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a precise document QA assistant. Answer strictly based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return answer, retrieved_texts

if __name__ == "__main__":
    query = "What is the primary topic of this document?"
    print(f"\nQuery: {query}\n")
    answer, sources = query_rag(query)
    print(f"Answer:\n{answer}\n")
    print(f"Retrieved Chunk Sample:\n{sources[0][:150]}...")