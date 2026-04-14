import streamlit as st
from recommender import UserBasedRecommender

st.title("🎬 Movie Recommendation System")

rec = UserBasedRecommender("data/ratings.csv")

user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Get Recommendations"):
    results = rec.recommend(user_id)

    st.subheader("Recommended Movies:")
    for movie in results:
        st.write(movie)
