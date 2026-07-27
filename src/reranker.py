from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Downloads/loads a lightweight local cross-encoder model."""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str], top_n: int = 3) -> List[str]:
        """Reranks candidates based on deep sentence pair relevance."""
        if not candidates:
            return []
        
        # Form (query, candidate) pairs for the cross-encoder
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.model.predict(pairs)
        
        # Sort candidates by descending score
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Return top_n reranked chunks
        return [doc for doc, score in scored_candidates[:top_n]]