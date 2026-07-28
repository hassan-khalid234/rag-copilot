# RAG Copilot with Hybrid Search & Reranking
 
A modular, framework-free Retrieval-Augmented Generation (RAG) system built in Python, combining dense and sparse retrieval with cross-encoder reranking to address precision failures common in naive vector search.
 
## Overview
 
Most basic RAG implementations rely on dense vector similarity alone, which struggles with exact keyword matches, jargon, and short factual queries. This project addresses that with a two-stage retrieval pipeline: hybrid search (dense + sparse) to widen the candidate pool, followed by cross-encoder reranking to select the most relevant chunks before they reach the LLM.
 
Built directly on core APIs rather than a framework like LangChain, to keep full control over how retrieval, ranking, and generation are wired together.
 
## Key Features
 
- **Framework-free architecture** — no LangChain or similar abstraction layer; retrieval logic is explicit and inspectable end to end
- **Hybrid search retrieval** — combines dense vector search (ChromaDB + `all-MiniLM-L6-v2`) with sparse keyword search (BM25) to catch both semantic intent and exact terminology
- **Two-stage cross-encoder reranking** — over-fetches candidates from hybrid search, then reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2` before passing context to the LLM
- **Fast generation** — powered by the Groq API running `llama-3.1-8b-instant`
- **Interactive UI** — Streamlit dashboard with real-time source chunk highlighting and latency tracing
## Architecture
 
```
[PDF Document]
      │
      ▼
[PyMuPDF Text Extraction]
      │
      ▼
[Sliding Window Chunker]
      │
      ├───► [Dense Vector Index (ChromaDB)]  ──┐
      │                                        ├──► [Candidate Pool (top-10)]
      └───► [Sparse Keyword Index (BM25)]    ──┘                 │
                                                                  ▼
                                                   [Cross-Encoder Reranker]
                                                                  │
                                                                  ▼
                                                    [Top 3 Reranked Contexts]
                                                                  │
                                                                  ▼
                                                    [Groq Llama 3.1 8B LLM]
                                                                  │
                                                                  ▼
                                                     [Answer + Citations UI]
```
 
## Why Hybrid Search + Reranking
 
- **Dense-only retrieval** misses exact keyword and jargon matches that don't cluster well in embedding space — a common failure mode for technical or domain-specific documents.
- **BM25** catches those exact-term matches, so combining it with dense search widens recall before anything gets filtered out.
- **Cross-encoder reranking** scores the query against each candidate chunk jointly (rather than comparing precomputed embeddings), which is more accurate than raising `top-k` alone — reranking fixes precision *after* recall, rather than just returning more unranked chunks and hoping the LLM sorts it out.
## Tech Stack
 
- **Retrieval:** ChromaDB, `all-MiniLM-L6-v2`, BM25, `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Extraction:** PyMuPDF
- **LLM:** Groq API (`llama-3.1-8b-instant`)
- **UI:** Streamlit
## Setup
 
```bash
git clone https://github.com/hassan-khalid234/rag-copilot.git
cd rag-copilot
pip install -r requirements.txt
```
 
Add your Groq API key to a `.env` file:
```
GROQ_API_KEY=your_key_here
```
 
## Usage
 
1. Place your PDF document(s) in the `data/` folder 
2. Run the ingestion step to build the dense and sparse indexes
3. Launch the app:
```bash
streamlit run streamlit_app.py
```
 
 
## Project Structure
 
```
├── data/                   # source PDFs
├── src/                # hybrid search + reranking logic
├── streamlit_app.py
├── requirements.txt
└── .env.example
```
 
## Status
 
Actively developed as a personal project exploring production-grade retrieval techniques beyond naive vector search.
 
## Author
 
Muhammad Hassan — [LinkedIn](https://www.linkedin.com/in/hassan-khalid-4028842a1/)
