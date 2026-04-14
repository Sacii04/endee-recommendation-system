# 🎬 End-to-End Movie Recommendation System

## 📌 Overview
This project is a **User-Based Collaborative Filtering Recommendation System** built using Python and Streamlit.  
It recommends movies to users based on the preferences of similar users.

The system uses **cosine similarity** to identify users with similar tastes and suggests movies they liked.

---

## 🚀 Features
- User-Based Collaborative Filtering  
- Cosine Similarity for user comparison  
- Interactive UI using Streamlit  
- Evaluation metrics (Precision & Recall)  
- Clean and modular project structure  

---

## 🧠 How It Works
1. Create a **user-item matrix** from ratings data  
2. Compute similarity between users using cosine similarity  
3. Identify most similar users  
4. Recommend movies liked by similar users but not watched by the target user  

---

## 📊 Dataset
The dataset contains user ratings for movies:

| user_id | movie_title | rating |
|--------|------------|--------|
| 1 | Inception | 5 |
| 2 | Avengers | 4 |

You can replace this dataset with larger datasets like MovieLens for better results.

---

## 📁 Project Structure
endee-recommendation-system/
│
├── data/
│   └── ratings.csv
│
├── recommender.py
├── evaluate.py
├── app.py
├── requirements.txt
└── README.md

## ⚙️ Installation & Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the app
streamlit run app.py

### 3. Open in browser
http://localhost:8501
