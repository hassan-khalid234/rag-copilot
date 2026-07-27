import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorStoreManager:
    def __init__(self, collection_name: str = "naive_rag_collection"):
        # Uses default sentence-transformers (all-MiniLM-L6-v2) under the hood
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.embedding_fn
        )

    def add_documents(self, chunks: List[str]):
        """Embeds and indexes text chunks."""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids
        )
        print(f"✅ Successfully indexed {len(chunks)} chunks into ChromaDB.")

    def query_similarity(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieves top_k nearest neighbors using Cosine Similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        if results['documents'] and results['documents'][0]:
            return results['documents'][0]
        return []