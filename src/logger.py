import time
from typing import List, Dict, Any

class RAGLogger:
    @staticmethod
    def log_step(step_name: str, details: str):
        print(f"\n[TRACE - {step_name.upper()}] {details}")

    @staticmethod
    def log_retrieval(query: str, raw_hits: int, reranked_chunks: List[str], duration_ms: float):
        print("\n" + "="*50)
        print(f"🔍 QUERY: {query}")
        print(f"⏱️ RETRIEVAL TIME: {duration_ms:.2f} ms")
        print(f"📦 CANDIDATE POOL: {raw_hits} chunks fetched")
        print(f"🎯 TOP RERANKED CHUNKS ({len(reranked_chunks)}):")
        for i, chunk in enumerate(reranked_chunks, 1):
            snippet = chunk.replace('\n', ' ')[:100]
            print(f"   [{i}] {snippet}...")
        print("="*50)