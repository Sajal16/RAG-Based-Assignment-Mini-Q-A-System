# RAG-Based-Assignment-Mini-Q-A-System
# 📄 Chat with PDF using RAG (Gemini + LlamaIndex + FAISS)

## 🚀 Project Overview
This application allows users to ask questions from an uploaded pdf. It used RAG (Retrieval Augmented Generation). It follows a step-by-step architecture to retrieve information from pdf and generate answers.


## 🧠 Tech Stack
- LlamaIndex (RAG Framework)
- Gemini API (LLM)
- HuggingFace (Embeddings)
- FAISS (Vector Database)
- Streamlit (UI)


## ⚙️ Features
- Upload PDF
- Ask questions in chat format
- Fast retrieval using FAISS
- Source-based answers (no hallucination)
- Chat history


## Architecture
PDF → Chunking → Embeddings → FAISS → Retrieval → Gemini → Answer


## 📦 Installation

```bash
pip install -r requirements.txt

```bash
streamlit run app.py
