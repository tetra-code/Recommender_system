import csv

import numpy as np
import pandas as pd

import utility


def find_biases(
        ratings_data: np.ndarray,
        global_avg_rating: float,
        user_index: int,
        movie_index: int,
) -> (float, float):
    movie_avg_rating = utility.movie_average_rating(movie_index, ratings_data)
    user_avg_rating = utility.user_average_rating(user_index, ratings_data)
    movie_bias = movie_avg_rating - global_avg_rating
    user_bias = user_avg_rating - global_avg_rating
    result = global_avg_rating + movie_bias + user_bias
    assert movie_avg_rating + user_avg_rating - global_avg_rating == result
    return movie_bias, user_bias


def find_latent_factors():
    return -2


def main():
    # they are all initially strings
    movie_data = utility.import_data('resources/movies.csv', ';')
    users_data = utility.import_data('resources/users.csv', ';')
    ratings_data = utility.import_data('resources/ratings.csv', ';')
    prediction_data = utility.import_data('resources/predictions.csv', ';')

    # 3.58131
    global_avg_rating = utility.all_movies_average_rating(ratings_data)

