import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.chat_models import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from typing import List, Dict

class RAGEvaluator:
    def __init__(self):
        # RAGAS uses LangChain LLM wrappers under the hood to judge quality
        self.eval_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def evaluate_sample(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str], 
        ground_truth: str = ""
    ) -> Dict[str, float]:
        """Evaluates a single RAG response across key RAGAS metrics."""
        
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth] if ground_truth else [""]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Run standard evaluation suite
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=self.eval_llm
        )
        
        return results