import csv
import random

import numpy as np
import pandas as pd


def import_data(file_path: str, delimiter: str) -> np.ndarray:
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        return np.array([row for row in reader])


"""
overall movies mean rating
"""
def all_movies_average_rating(ratings_data: np.ndarray) -> float:
    # extract only ratings and covert str to float
    ratings = ratings_data[:, 2].astype(float)
    return float(np.mean(ratings))


"""
Calculates the similarity value by subtracting the avg movie ratings and computing the
cosine similarity for each listed movie. Output is in range of [-1, 1]

Dataframe has columns user_idx, rating_x, rating_y

"""
def pearson_correlation_coefficient(
        common_users: pd.DataFrame,
) -> float:
    denominator_lhs = common_users['rating_x'].apply(lambda x: np.power(x, 2)).sum()
    denominator_rhs = common_users['rating_y'].apply(lambda y: np.power(y, 2)).sum()
    numerator = (common_users['rating_x'] * common_users['rating_y']).sum()
    denominator = np.sqrt(denominator_lhs) * np.sqrt(denominator_rhs)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


"""
returns nested np array of user index and the rating - avg_rating for specified movie

rating_data has user_index, movie_index, rating in column order
"""
def find_users_rated_movie_min_avg(
        movie_index: int,
        ratings_data: np.ndarray,
        avg_rating: float,
) -> pd.DataFrame:
    user_rating_df = [
        [int(row[0]), float(row[2]) - avg_rating]
        for row in ratings_data if int(row[1]) == movie_index
    ]
    return pd.DataFrame(user_rating_df, columns=['user_idx', 'rating'])


"""
Part of item-item collaborative filtering to find k neighbours of movie x

Index of x is inserted in beginning of return list for simplicity for write
"""
def find_k_neighbours(
        x_movie_index: int,
        movies_len: int,
        movie_avg_ratings_data: np.ndarray,
        ratings_data: np.ndarray,
        similarity_threshold: float,
        k: int,
) -> list:
    result = [x_movie_index]
    x_movie_avg_rating = float(movie_avg_ratings_data[x_movie_index - 1][1])
    attempts = 0
    # already minus the avg rating for convenience when pearson coefficient calculation
    df_users_movie_x_minus_avg_rating = find_users_rated_movie_min_avg(
        x_movie_index,
        ratings_data,
        x_movie_avg_rating,
    )

    r = list(range(1, movies_len+1))
    random.shuffle(r)
    for i in r:
        if (attempts == 300):
            return result
        attempts += 1
        y_movie_index = i
        if y_movie_index == x_movie_index:
            continue
        y_movie_avg_rating = float(movie_avg_ratings_data[y_movie_index - 1][1])

        # already minus the avg rating for convenience when pearson coefficient calculation
        df_users_movie_y_minus_avg_rating = find_users_rated_movie_min_avg(
            y_movie_index,
            ratings_data,
            y_movie_avg_rating,
        )

        # merge and g22et common users data frame with columns user_idx, rating_x, rating_y
        common_users = pd.merge(df_users_movie_x_minus_avg_rating, df_users_movie_y_minus_avg_rating, on="user_idx")
        similarity_value = pearson_correlation_coefficient(common_users)
        if similarity_value >= similarity_threshold:
            result.append(y_movie_index)
            if len(result) == k:
                return result
    return result
