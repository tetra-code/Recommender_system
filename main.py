from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split

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
    param_grid = {
        'n_factors': [20, 50, 100],
        'n_epochs': [5, 10, 20],
    }
    gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=10)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    # best hyper-parameters
    best_factor = gs.best_params['rmse']['n_factors']
    best_epoch = gs.best_params['rmse']['n_epochs']

    svd = SVD(n_factors=best_factor, n_epochs=best_epoch)
    return svd


def train(svd, data):
    # Train the algorithm on the trainset
    trainset, _ = train_test_split(data, test_size=.20)
    svd.fit(trainset)

    # Run 10-fold cross-validation and print results
    results = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=10)
    print(results)


"""
# define which user ID that we want to give recommendation
userID = 23
# define how many top-n movies that we want to recommend
n_items = 10
# generate recommendation using the model that we have trained
generate_recommendation(svd, userID, ratings_data, movies_data, n_items)
"""
def predict_rating(model, user_id, ratings_df, movies_df, n_items):
    # Get a list of all movie IDs from dataset
    movie_ids = ratings_df["movieId"].unique()

    # Get a list of all movie IDs that have been watched by user
    movie_ids_user = ratings_df.loc[ratings_df["userId"] == user_id, "movieId"]
    # Get a list off all movie IDS that that have not been watched by user
    movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)

    # Apply a rating of 4 to all interactions (only to match the Surprise dataset format)
    test_set = [[user_id, movie_id, 4] for movie_id in movie_ids_to_pred]

    # Predict the ratings and generate recommendations
    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    print("Top {0} item recommendations for user {1}:".format(n_items, user_id))
    # Rank top-n movies based on the predicted ratings
    index_max = (-pred_ratings).argsort()[:n_items]
    for i in index_max:
        movie_id = movie_ids_to_pred[i]
        print(movies_df[movies_df["movieId"] == movie_id]["title"].values[0], pred_ratings[i])


def main():
    # they are all initially strings

    ratings_data = pd.read_csv(
        'resources/ratings.csv',
        sep=';',
        names=['user_idx', 'movie_idx', 'rating'],
    )
    user_metadata = pd.read_csv(
        'resources/users.csv',
        sep=';',
        names=['user_idx', 'sex', 'age', 'profession'],
    )

    data = get_ratings_data(ratings_data)
    svd_model = grid_search_cross_vali_svd(data)
    train(svd_model, data)

    # Apply a rating of 4 to all interactions (only to match the Surprise dataset format)
    prediction_data = utility.import_data('resources/predictions.csv', ';')
    test_set = [[ int(row[0]), int(row[1]), 0 ] for row in prediction_data]
    predictions = svd_model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])


train()