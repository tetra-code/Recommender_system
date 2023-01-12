import csv

from surprise import Dataset, SVD, Reader, SVDpp, CoClustering, BaselineOnly, KNNWithZScore, SlopeOne, NMF, \
    NormalPredictor, KNNBasic, KNNWithMeans, KNNBaseline
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


def find_best_algo(data):
    benchmark = []
    for algorithm in [SVD(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(),
                      KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # Perform cross validation
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)

    print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse'))


"""
Use search cross-validation for hyperparamter tuning, gets the best number of 
factors and number of epochs to choose from
"""
def search_cross_vali_svd(search, data):
    # List of n factors and epochs to choose from
    search.fit(data)
    print(search.best_score['rmse'])
    print(search.best_params['rmse'])

    # best hyper-parameters
    best_factor = search.best_params['rmse']['n_factors']
    best_epoch = search.best_params['rmse']['n_epochs']
    best_lr_all = search.best_params['rmse']['lr_all']
    best_reg_all = search.best_params['rmse']['reg_all']
    return SVDpp(
        n_factors=best_factor,
        n_epochs=best_epoch,
        lr_all=best_lr_all,
        reg_all=best_reg_all,
        cache_ratings=True,
    )


def train_and_validate(algo, data) -> bool:
    # Train the algorithm on the training set
    train_set, _ = train_test_split(data, test_size=.2)
    algo.fit(train_set)
    print('training done')
    # Run 5-fold cross-validation and print results
    scores = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5)
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

    # so far best is n_factors=10, n_epochs=100, lr_all=0.003, reg_all=0.03,
    param_grid = {
        'n_factors': [3, 10, 20],
        'n_epochs': [50, 100],
        "lr_all": [0.002, 0.003, 0.005, 0.006, 0.007],
        "reg_all": [0.01, 0.02, 0.03, 0.4, 0.5]
    }
    rs = RandomizedSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=5, refit=True, joblib_verbose=2, random_state=42)
    svd_model = search_cross_vali_svd(rs, data)

    # train
    print('training')
    if not train_and_validate(svd_model, data):
        print("not good enough")
        return

    # Apply a rating of 0 to all interactions (only to match the Surprise dataset format)
    prediction_data = utility.import_data('resources/predictions.csv', ';')
    test_set = [[ int(row[0]), int(row[1]), 0 ] for i, row in enumerate(prediction_data)]

    # list of Prediction instance
    predictions = svd_model.test(test_set)

    # only extract the prediction (rating estimates)
    # pred_ratings = np.array([int(round(pred.est)) for pred in predictions])
    pred_ratings = np.array([pred.est for pred in predictions])

    # write the predictions in submission file
    with open('resources/submission.csv', 'w', newline='') as submission_file:
        submission_writer = csv.writer(submission_file, delimiter=',')
        submission_writer.writerow(['Id', 'Rating'])
        for i, pred_rating in enumerate(pred_ratings):
            submission_writer.writerow([i+1, pred_rating])
    print('Finished')


main()
