import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedRecommender:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.user_movie_matrix = None
        self.similarity = None
        self._prepare()

    def _prepare(self):
        self.user_movie_matrix = self.df.pivot_table(
            index='user_id',
            columns='movie_title',
            values='rating'
        ).fillna(0)

        self.similarity = cosine_similarity(self.user_movie_matrix)

    def recommend(self, user_id, top_n=3):
        if user_id not in self.user_movie_matrix.index:
            return ["User not found"]

        user_index = list(self.user_movie_matrix.index).index(user_id)

        sim_scores = list(enumerate(self.similarity[user_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        similar_users = sim_scores[1:3]

        recommendations = set()

        for user, _ in similar_users:
            user_movies = self.user_movie_matrix.iloc[user]
            liked_movies = user_movies[user_movies >= 4].index
            recommendations.update(liked_movies)

        watched = self.user_movie_matrix.loc[user_id]
        watched_movies = watched[watched > 0].index

        final = [movie for movie in recommendations if movie not in watched_movies]

        return final[:top_n]
