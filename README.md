# 🎬 AI-Powered Hybrid Recommendation System using Semantic Search and RAG

## 📌 Overview
This project is an advanced AI-driven recommendation system that combines:
- Semantic Search using embeddings
- Hybrid Recommendation (Content-based + Collaborative Filtering)
- Retrieval-Augmented Generation (RAG) for explainable recommendations

## 🚀 Key Features
- Semantic Search
- Hybrid Recommendation
- RAG-based explanations
- Evaluation metrics (Precision@K, Recall@K)

## 🧠 Architecture
User Query → Embedding → FAISS → Retrieval → LLM → Response

## 🛠️ Tech Stack
- FAISS
- SentenceTransformers
- OpenAI GPT
- Pandas, NumPy

## 📂 Project Structure
ai-recommendation-system/
├── data/
├── recommender.py
├── app.py
├── evaluate.py
├── requirements.txt
└── README.md

## ⚙️ Installation
pip install -r requirements.txt

## ▶️ Run
python app.py

Built an AI-powered hybrid recommendation system using semantic search and RAG.
