from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def fit(self, chunks: List[str]):
        """Tokenizes chunks and initializes the BM25 index."""
        self.chunks = chunks
        tokenized_corpus = [doc.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Scores documents based on keyword matching."""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip chunks with scores and sort descending
        chunk_scores = list(zip(self.chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score in chunk_scores[:top_k]:
            if score > 0:  # Only include non-zero keyword matches
                results.append({"chunk": chunk, "score": float(score)})
        return results