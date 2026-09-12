import os
from groq import Groq
from typing import List

class RAGGenerator:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.client = Groq()  # Automatically picks up GROQ_API_KEY from .env
        self.model_name = model_name

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        context_block = "\n---\n".join(context_chunks)
        
        prompt = f"""
Answer the user query based ONLY on the provided context below.
If the answer cannot be found in the context, say "I cannot find this information in the document."

Context:
{context_block}

Query: {query}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise technical document assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content