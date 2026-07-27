# 🤖 Production-Grade RAG Copilot

A modular, framework-free Retrieval-Augmented Generation (RAG) system built in Python. Designed to demonstrate production-level retrieval techniques, including **Hybrid Search (Dense + BM25)** and **Cross-Encoder Reranking**, eliminating common precision failures found in naive vector search.

---

## 🌟 Key Features

* **Vanilla Python Architecture:** Zero reliance on heavy abstractions like LangChain—built directly on core APIs for full control over retrieval mechanics.
* **Hybrid Search Retrieval:** Combines **Dense Vectors** (ChromaDB + `all-MiniLM-L6-v2`) with **Sparse Keyword Search** (`BM25`) to catch both semantic intent and exact jargon matches.
* **Two-Stage Cross-Encoder Reranking:** Over-fetches candidate chunks and filters them using `cross-encoder/ms-marco-MiniLM-L-6-v2` before passing context to the LLM.
* **Ultra-Fast Generation:** Powered by **Groq API** running `llama-3.1-8b-instant`.
* **Interactive UI:** Streamlit dashboard featuring real-time source chunk highlighting and latency tracing.

---

## 🏗️ Architecture Pipeline

```text
[PDF Document]
      │
      ▼
[PyMuPDF Text Extraction]
      │
      ▼
[Sliding Window Chunker]
      │
      ├───► [Dense Vector Index (ChromaDB)]  ──┐
      │                                       ├──► [Candidate Pool (top-10)]
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

---

## 🧪 Step 4: Test the Streamlit App

Run the app locally to make sure everything operates smoothly:

```powershell
streamlit run streamlit_app.py
