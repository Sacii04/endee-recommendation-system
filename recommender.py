import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/items.csv")
ratings = pd.read_csv("data/ratings.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['description'].tolist())

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def semantic_search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]

user_item_matrix = ratings.pivot_table(index='userId', columns='itemId', values='rating').fillna(0)

def collaborative_filter(user_id, top_k=5):
    if user_id not in user_item_matrix.index:
        return None
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_item_matrix.corrwith(user_ratings, axis=1).sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    top_users = similar_users.head(5).index
    recommended_items = user_item_matrix.loc[top_users].mean().sort_values(ascending=False)
    return df[df['itemId'].isin(recommended_items.head(top_k).index)]

def hybrid_recommend(query, user_id=None, top_k=5):
    semantic_results = semantic_search(query, top_k)
    if user_id:
        collab_results = collaborative_filter(user_id, top_k)
        if collab_results is not None:
            return pd.concat([semantic_results, collab_results]).drop_duplicates().head(top_k)
    return semantic_results
