import csv

import numpy as np

import utility as util


"""
Write to movie_avg_rating.csv

1st column: movie index
2nd column: avg rating
"""
def write_user_avg_rating():
    user_data = util.import_data('resources/users.csv', ';')
    ratings_data = util.import_data('resources/ratings.csv', ';')
    with open('resources/user_avg_rating.csv', 'w', newline='') as avg_rating_file:
        avg_rating_writer = csv.writer(avg_rating_file, delimiter=';')
        for row in user_data:
            user_index = int(row[0])
            avg_rating = float( np.mean([float(row[2]) for row in ratings_data if int(row[0]) == user_index]) )
            avg_rating_writer.writerow([user_index, round(avg_rating, 5)])


"""
Write to movie_avg_rating.csv

1st column: movie index
2nd column: avg rating
"""
def write_movie_avg_ratings():
    movie_data = util.import_data('resources/movies.csv', ';')
    ratings_data = util.import_data('resources/ratings.csv', ';')
    with open('resources/movie_avg_rating.csv', 'w', newline='') as avg_rating_file:
        avg_rating_writer = csv.writer(avg_rating_file, delimiter=';')
        for row in movie_data:
            movie_index = int(row[0])
            avg_rating = float( np.mean([float(row[2]) for row in ratings_data if int(row[1]) == movie_index]) )
            avg_rating_writer.writerow([movie_index, round(avg_rating, 5)])



def write_k_neighbour():
    movie_avg_ratings_data = util.import_data('resources/movie_avg_rating.csv', ';')
    ratings_data = util.import_data('resources/ratings.csv', ';')
    movie_data = util.import_data('resources/movies.csv', ';')
    movies_len = len(movie_data)
    with open('resources/k_neighbours.csv', 'a', newline='') as k_neighbours_file:
        k_neighbours_writer = csv.writer(k_neighbours_file, delimiter=';')
        for i in range(126, movies_len+1):
            movie_x_and_k_neighbours = util.find_k_neighbours(
                x_movie_index=i,
                movies_len=movies_len,
                movie_avg_ratings_data=movie_avg_ratings_data,
                ratings_data=ratings_data,
                similarity_threshold=0.5,
                k=10,
            )
            k_neighbours_writer.writerow(movie_x_and_k_neighbours)
            print(movie_x_and_k_neighbours)


write_k_neighbour()