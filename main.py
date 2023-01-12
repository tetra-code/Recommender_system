import csv

from surprise import Dataset, SVD, Reader, SVDpp
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split, RandomizedSearchCV

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


def get_ratings_data(ratings_data):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(
        ratings_data[['user_idx', 'movie_idx', 'rating']],
        reader,
    )


"""
Use grid search cross-validation for hyperparamter tuning, gets the best number of 
factors and number of epochs to choose from
"""
def grid_search_cross_vali_svd(data):
    # List of n factors and epochs to choose from
    print("Starting grid search")
    param_grid = {
        'n_factors': [10, 20, 30],
        'n_epochs': [30, 50],
    }
    gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=5, joblib_verbose=2)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    # best hyper-parameters
    best_factor = gs.best_params['rmse']['n_factors']
    best_epoch = gs.best_params['rmse']['n_epochs']

    return SVD(n_factors=best_factor, n_epochs=best_epoch, biased=True)


"""
Use grid search cross-validation for hyperparamter tuning, gets the best number of 
factors and number of epochs to choose from
"""
def rand_search_cross_vali_svd(data):
    # List of n factors and epochs to choose from
    print("Starting grid search")
    param_grid = {
        'n_factors': [10, 20, 30],
        'n_epochs': [50, 100],
    }
    rs = RandomizedSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=5, n_iter=5, joblib_verbose=2, random_state=42)
    rs.fit(data)
    print(rs.best_score['rmse'])
    print(rs.best_params['rmse'])

    # best hyper-parameters
    best_factor = rs.best_params['rmse']['n_factors']
    best_epoch = rs.best_params['rmse']['n_epochs']

    return SVD(n_factors=best_factor, n_epochs=best_epoch, biased=True)


def train_and_validate(svd, data) -> bool:
    # Train the algorithm on the training set
    train_set, _ = train_test_split(data, test_size=.20)
    svd.fit(train_set)
    print('training done')

    # Run 10-fold cross-validation and print results
    scores = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5)
    for score in scores['test_rmse']:
        print(score)
        if score > 0.88:
            return False
    return True


def main():
    # they are all initially strings
    ratings_data = pd.read_csv(
        'resources/ratings.csv',
        sep=';',
        names=['user_idx', 'movie_idx', 'rating'],
    )

    data = get_ratings_data(ratings_data)
    svd_model = rand_search_cross_vali_svd(data)
    # svd_model = SVD(n_factors=10, n_epochs=100, biased=True)
    print('training')
    if not train_and_validate(svd_model, data):
        print("not good enough")
        return

    # Apply a rating of 4 to all interactions (only to match the Surprise dataset format)
    prediction_data = utility.import_data('resources/predictions.csv', ';')
    test_set = [[ int(row[0]), int(row[1]), 4] for i, row in enumerate(prediction_data)]

    # list of Prediction instance
    predictions = svd_model.test(test_set)

    # only extract the prediction (rating estimates)
    pred_ratings = np.array([int(round(pred.est)) for pred in predictions])

    # write the predictions in submission file
    with open('resources/submission.csv', 'w', newline='') as submission_file:
        submission_writer = csv.writer(submission_file, delimiter=',')
        for i, pred_rating in enumerate(pred_ratings):
            print(i+1)
            submission_writer.writerow([i+1, pred_rating])


main()
