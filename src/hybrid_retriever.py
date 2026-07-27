from typing import List
from src.vector_store import VectorStoreManager
from src.sparse_search import BM25Retriever
from src.reranker import Reranker

class HybridRAGRetriever:
    def __init__(self, vector_db: VectorStoreManager):
        self.vector_db = vector_db
        self.bm25_retriever = BM25OkapiRetriever() if hasattr(self, 'BM25OkapiRetriever') else BM25Retriever()
        self.reranker = Reranker()

    def index_documents(self, chunks: List[str]):
        """Indexes chunks into both Dense (ChromaDB) and Sparse (BM25) stores."""
        self.vector_db.add_documents(chunks)
        self.bm25_retriever.fit(chunks)

    def retrieve(self, query: str, fetch_k: int = 10, final_top_k: int = 3) -> List[str]:
        # 1. Fetch top_k from Dense Vector DB
        dense_results = self.vector_db.query_similarity(query, top_k=fetch_k)
        
        # 2. Fetch top_k from Sparse BM25
        sparse_hits = self.bm25_retriever.search(query, top_k=fetch_k)
        sparse_results = [hit["chunk"] for hit in sparse_hits]

        # 3. Merge & Deduplicate candidates
        candidate_pool = list(set(dense_results + sparse_results))
        
        if not candidate_pool:
            return []

        # 4. Two-Stage Reranking via Cross-Encoder
        final_chunks = self.reranker.rerank(query, candidate_pool, top_n=final_top_k)
        return final_chunks