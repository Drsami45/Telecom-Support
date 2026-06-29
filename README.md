# 📡 Telecom Customer Support Chatbot (RAG)

A production-ready RAG chatbot for telecom customer support, powered by Qwen3-32B on Groq.

## Features
- Multi-source retrieval: FAQ (CSV), Support Tickets (SQLite), PDF Guide
- Source citations in every answer
- Confidence-based fallback (skips LLM if no relevant docs found)
- Retrieval evaluation script (Recall@3)
- Streamlit chat UI with real-time streaming

## Tech Stack
Python · LangChain · ChromaDB · HuggingFace Embeddings · Groq API · Streamlit

## Setup

1. Clone the repo
   git clone https://github.com/Drsami45/Telecom-Support.git
   cd telecom-rag-chatbot

2. Install dependencies
   pip install -r requirements.txt

3. Add your API key — create a .env file:
   GROQ_API_KEY=your_key_here

4. Run ingestion scripts
   python ingest_faq.py
   python ingest_pdf_copy.py
   python ingest_tickets.py

5. Launch the app
   streamlit run app.py
