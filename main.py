import csv

import numpy as np

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


"""
To ask:

does ratings data have to be separated to train and test when runing latent factors

Create ghost row and column around ratings matrix since user and movie index based
on 1-index, not 0-index

"""
def find_latent_factors(
        num_users: int,
        num_user_factors: int,
        num_movies: int,
        num_movie_factors: int,
        alpha: float,
        epochs: int,
        ratings_data: np.ndarray,
) -> np.ndarray:
    # P is user factor matrix (later needs to be transformed when multiplying)
    P = np.random.normal(0, .1, (num_users+1, num_user_factors))

    # Q is movie factor matrix
    Q = np.random.normal(0, .1, (num_movies+1, num_movie_factors))

    for epoch in range(epochs):
        for row in ratings_data:
            x = int(row[0])
            i = int(row[1])
            r_xi = float(row[2])
            residual = r_xi - np.dot(P[x], Q[i])

            # Stochastic gradient
            # we want to update them at the same time, so we make a temporary variable.
            # TODO: here also need to update bias matrix at each stochastic gradient descent
            temp = P[x, :]
            P[x, :] += alpha * residual * Q[i]
            Q[i, :] += alpha * residual * temp

    ratings_matrix = P.T * Q
    return ratings_matrix


def main():
    # they are all initially strings
    movie_data = utility.import_data('resources/movies.csv', ';')
    users_data = utility.import_data('resources/users.csv', ';')
    ratings_data = utility.import_data('resources/ratings.csv', ';')
    prediction_data = utility.import_data('resources/predictions.csv', ';')

    # 3.58131
    global_avg_rating = utility.all_movies_average_rating(ratings_data)

    # TODO: create bias matrix that needs to be updated at each gradient descent?
    # baseline bias = global avg rating + user x deviation + movie i deviation
    # bxi = u + bx + bi

    # find ratings matrix using SVP and stochastic gradient descent
    # ratings data has dummy column and dummy row since user and movie indices start at 1
    ratings_matrix = find_latent_factors(
        num_users=6040,
        num_user_factors=3,
        num_movies=3706,
        num_movie_factors=2,
        alpha=0.3,
        epochs=300,
        ratings_data=ratings_data,
    )

    # prediction
    for row in prediction_data:
        user_idx = int(row[0])
        movie_idx = int(row[1])

        # remove 1 for each index since index in csv is 1-based
        pred_rating = ratings_matrix[user_idx][movie_idx]
