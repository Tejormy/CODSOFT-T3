import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample data
def load_data():
    data = {
        "UserID": ["585", "5c2", "590", "5b3", "5b0"],
        "MovieID": [10, 20, 30, 40, 50],
        "Rating": [5, 4, 2, 4, 5],
    }

    ratings = pd.DataFrame(data)
    movies = pd.DataFrame({
        "MovieID": [10, 20, 30, 40, 50],
        "Title": ["The World", "Ironman", "Deadpool", "The Dark Knight", "Inception"]
    })

    return ratings, movies

# Content-Based Filtering Recommendation Function
def recommend_movies_content_based(user_id, ratings, movies, num_recommendations=2):
    user_ratings = ratings[ratings["UserID"] == user_id]
    avg_ratings = ratings.groupby("MovieID")["Rating"].mean()
    
    recommendations = avg_ratings[~avg_ratings.index.isin(user_ratings["MovieID"])].sort_values(ascending=False)
    recommended_movies = pd.merge(recommendations.head(num_recommendations).reset_index(), movies, on="MovieID")
    
    return recommended_movies[["Title", "Rating"]]

# Collaborative Filtering Recommendation Function
def recommend_movies_collaborative(user_id, user_movie_matrix, user_similarity_df, movies, num_recommendations=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users_ratings = user_movie_matrix.loc[similar_users.index]
    weighted_ratings = similar_users_ratings.T.dot(similar_users) / similar_users.sum()
    
    user_rated_movies = user_movie_matrix.loc[user_id].dropna().index
    recommendations = weighted_ratings[~weighted_ratings.index.isin(user_rated_movies)].sort_values(ascending=False)
    
    recommended_movies = pd.merge(recommendations.head(num_recommendations).reset_index(), movies, on="MovieID")
    
    return recommended_movies[["Title", 0]]

def main():
    ratings, movies = load_data()
    
    user_movie_matrix = ratings.pivot_table(index="UserID", columns="MovieID", values="Rating")
    
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    
    user_similarity = cosine_similarity(user_movie_matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    user_id = input("Enter User ID: ").strip()
    method = input("Choose recommendation method (content-based / collaborative): ").strip().lower()

    if method == "content-based":
        recommendations = recommend_movies_content_based(user_id, ratings, movies)
    elif method == "collaborative":
        recommendations = recommend_movies_collaborative(user_id, user_movie_matrix, user_similarity_df, movies)
    else:
        print("Invalid method selected!")
        return

    print(f"\nRecommendations for User {user_id} using {method} filtering:")
    print(recommendations)

if __name__ == "__main__":
    main()
