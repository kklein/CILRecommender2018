import copy
import os
from datetime import datetime
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_FILE = os.path.join(ROOT_DIR, 'data/data_train.csv')
TRAINING_FILE_NAME = os.path.join(
    ROOT_DIR, 'data/trainingIndices.csv')
VALIDATION_FILE_NAME = os.path.join(
    ROOT_DIR, 'data/validationIndices.csv')
VALIDATION_MASK_FILE_NAME = os.path.join(
    ROOT_DIR, 'data/train_valid_80_10_10/validationIndices_mask.csv')
AUX = os.path.join(
    ROOT_DIR, 'data/train_valid_80_10_10/validationIndices_first.csv')
META_VALIDATION_FILE_NAME = os.path.join(
    ROOT_DIR, 'data/train_valid_80_10_10/validationIndices_second.csv')

SAMPLE_SUBMISSION = os.path.join(ROOT_DIR, \
            'data/sampleSubmission.csv')
ENSEMBLE_INPUT_DIR = 'data/stacking/good_data'
ITEM_COUNT = 1000
USER_COUNT = 10000
WEIGHT_KNN = 0.001
N_NEIGHBORS = 3
USER_COUNT_WEIGHT = 10
SAVE_META_PREDICTIONS = False

def load_ratings(data_file=DATA_FILE):
    ratings = []
    with open(data_file, 'r') as file:
        # Read header.
        _ = file.readline()
        for line in file:
            key, value_string = line.split(",")
            rating = float(value_string)
            row_string, col_string = key.split("_")
            row = int(row_string[1:])
            col = int(col_string[1:])
            ratings.append((row - 1, col - 1, rating))
    return ratings

def ratings_to_matrix(ratings):
    matrix_rows = USER_COUNT
    matrix_cols = ITEM_COUNT
    matrix = np.zeros([matrix_rows, matrix_cols])
    for row, col, rating in ratings:
        matrix[row, col] = rating
    return matrix

def mask_validation(data, use_three_way):
    masked_data = np.copy(data)
    if use_three_way:
        mask_file = VALIDATION_MASK_FILE_NAME
    else:
        mask_file = VALIDATION_FILE_NAME
    mask_indices = get_indices_from_file(mask_file)
    for row_index, col_index in mask_indices:
        masked_data[row_index][col_index] = 0
    return masked_data

def get_validation_indices(use_three_way):
    if use_three_way:
        validation_indices = get_indices_from_file(AUX)
    else:
        validation_indices = get_indices_from_file(VALIDATION_FILE_NAME)
    return validation_indices

def get_meta_validation_indices():
    return get_indices_from_file(META_VALIDATION_FILE_NAME)

def get_observed_indices(data):
    row_indices, col_indices = np.where(data != 0)
    return list(zip(row_indices, col_indices))

def get_unobserved_indices(data):
    row_indices, col_indices = np.where(data == 0)
    return list(zip(row_indices, col_indices))

def get_indices_from_file(file_name):
    indices = []
    with open(file_name, 'r') as file:
        # Read header.
        _ = file.readline()
        for line in file:
            i, j = line.split(",")
            indices.append((int(i), int(j)))
    return indices

def get_indices_to_predict():
    """Get list of indices to predict from sample submission file.
        Returns:
            indices_to_predict:  list of tuples with indices"""
    indices_to_predict = []
    with open(SAMPLE_SUBMISSION, 'r') as file:
        _ = file.readline()
        for line in file:
            key, _ = line.split(",")
            row_string, col_string = key.split("_")
            i = int(row_string[1:]) - 1
            j = int(col_string[1:]) - 1
            indices_to_predict.append((i, j))
    return indices_to_predict

def write_ratings(predictions, submission_file):
    with open(submission_file, 'w') as file:
        file.write('Id,Prediction\n')
        for i, j, prediction in predictions:
            file.write('r%d_c%d,%f\n' % (i, j, prediction))

def reconstruction_to_predictions(
        reconstruction, submission_file, indices_to_predict=None):
    if indices_to_predict is None:
        indices_to_predict = get_indices_to_predict()
    enumerate_predictions = lambda t: (
        t[0] + 1, t[1] + 1, reconstruction[t[0], t[1]])
    predictions = list(map(enumerate_predictions, indices_to_predict))
    write_ratings(predictions, submission_file)

def save_ensembling_predictions(reconstruction, name):
    reconstruction_to_predictions(
        reconstruction, ROOT_DIR + 'data/meta_training_' + name + '_stacking'
        + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=get_validation_indices(use_three_way=True))
    reconstruction_to_predictions(
        reconstruction, ROOT_DIR + 'data/meta_validation_' + name + '_stacking'
        + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=get_meta_validation_indices())

def clip(data):
    data[data > 5] = 5
    data[data < 1] = 1
    return data

def ampute_reconstruction(reconstruction, data):
    observed_indices = get_observed_indices(data)
    for row_index, col_index in observed_indices:
        reconstruction[row_index][col_index] = data[row_index][col_index]

def impute_by_avg(data, by_row):
    data = data.T if by_row else data
    for row in data:
        empty = (row == 0)
        row_sum = np.sum(row)
        row[empty] = row_sum / np.count_nonzero(row)
    return data.T if by_row else data

def impute_by_bias(data):
    total_average = np.mean(data[np.nonzero(data)])
    row_biases = np.zeros(data.shape[0])
    col_biases = np.zeros(data.shape[1])
    for row_index in range(data.shape[0]):
        row_biases[row_index] = np.sum(data[row_index]) / \
                np.count_nonzero(data[row_index]) - total_average
    for col_index in range(data.shape[1]):
        col_biases[col_index] = np.sum(data[:][col_index]) / \
                np.count_nonzero(data[:][col_index]) - total_average
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            if data[row_index, col_index] == 0:
                new_value = total_average + \
                        row_biases[row_index] + col_biases[col_index]
                data[row_index, col_index] = new_value
    return data

def impute_by_variance(data):
    global_average = np.sum(data) / np.count_nonzero(data)
    global_variance = np.var(data[data != 0])

    adjusted_movie_means = np.zeros((data.shape[1],))
    for i in range(data.shape[1]):
        movie_ratings = data[:, i]
        movie_ratings = movie_ratings[movie_ratings != 0]
        movie_variance = np.var(movie_ratings)
        relative_variance = movie_variance / global_variance
        adjusted_movie_means[i] = (
            global_average * relative_variance + np.sum(movie_ratings)) / (
                relative_variance + np.count_nonzero(movie_ratings))

    adjusted_user_deviation = np.zeros((data.shape[0],))
    for i in range(data.shape[0]):
        user_ratings = data[i]
        user_deviations = adjusted_movie_means - user_ratings
        user_deviations = user_deviations[user_ratings != 0]
        user_deviation_variance = np.var(user_deviations)
        relative_variance = user_deviation_variance / global_variance
        adjusted_user_deviation[i] = (
            global_average * relative_variance + sum(user_deviations)) / (
                relative_variance + np.count_nonzero(user_deviations))

    user_counts = np.count_nonzero(data, axis=1)
    movie_counts = np.count_nonzero(data, axis=0)

    movie_count_matrix = np.tile(movie_counts, (len(user_counts), 1))
    user_count_matrix = np.tile(user_counts, (len(movie_counts), 1)).T
    combined_matrix = copy.copy(
        movie_count_matrix) + USER_COUNT_WEIGHT * copy.copy(user_count_matrix)
    d_matrix = np.divide(movie_count_matrix, combined_matrix)

    m_matrix = np.tile(
        adjusted_movie_means, (len(adjusted_user_deviation), 1))
    u_matrix = np.tile(
        adjusted_user_deviation, (len(adjusted_movie_means), 1)).T

    data = np.multiply(m_matrix, d_matrix) + \
        np.multiply(u_matrix, np.ones(d_matrix.shape) - d_matrix)
    return data

def compute_rmse(data, prediction, indices=None):
    if indices is None:
        indices = get_indices_from_file(VALIDATION_FILE_NAME)
    squared_error = 0
    for i, j in indices:
        squared_error += (data[i][j] - prediction[i][j]) ** 2
    return np.sqrt(squared_error / len(indices))

def knn_smoothing(reconstruction, user_embeddings):
    normalized_user_embeddings = normalize(user_embeddings)
    knn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1)
    knn.fit(normalized_user_embeddings)
    distances, neighbors = knn.kneighbors(normalized_user_embeddings)
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]

    ones = np.ones(distances.shape)
    similarities = ones - distances
    weights = np.square(np.square(similarities))
    smoothed_data = np.zeros(reconstruction.shape)
    aggregated_neighbor_ratings = np.zeros(reconstruction.shape)

    for i in range(reconstruction.shape[0]):
        stacked_ratings = []
        for neighbor in neighbors[i]:
            stacked_ratings.append(reconstruction[neighbor])
        stacked_ratings = np.asarray(stacked_ratings)
        aggregated_neighbor_ratings[i] =\
                np.matmul(weights[i], stacked_ratings) / sum(weights[i])

    for i in range(reconstruction.shape[0]):
        smoothed_data[i] = (1 - WEIGHT_KNN) * reconstruction[i] + WEIGHT_KNN *\
                aggregated_neighbor_ratings[i]

    smoothed_data = clip(smoothed_data)
    return smoothed_data

def load_predictions_from_files(file_prefix='submission_'):
    path = os.path.join(ROOT_DIR, ENSEMBLE_INPUT_DIR)
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path, i)) and file_prefix in i]
    all_ratings = []
    for file in files:
        print("loading {}".format(file))
        ratings = load_ratings(file)
        ratings = ratings_to_matrix(ratings)
        all_ratings.append(ratings)
    return all_ratings

def compute_mean_predictions(all_ratings):
    reconstruction = np.mean(np.array(all_ratings), axis=0)
    reconstruction = impute_by_avg(reconstruction, by_row=False)
    return reconstruction
