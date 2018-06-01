import os
import numpy as np
import matplotlib.pyplot as plt
#import model_svd
import sklearn

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_FILE = os.path.join(ROOT_DIR, 'data/data_train.csv')
TRAINING_FILE_NAME = os.path.join(ROOT_DIR, \
            'data/trainingIndices.csv')
VALIDATION_FILE_NAME = os.path.join(ROOT_DIR, \
            'data/validationIndices.csv')

SAMPLE_SUBMISSION = os.path.join(ROOT_DIR, \
            'data/sampleSubmission.csv')
ITEM_COUNT = 1000
USER_COUNT = 10000

# https://stackoverflow.com/questions/42746248/numpy-linalg-norm-behaving-oddly-wrongly
def safe_norm(x):
    xmax = np.max(x)
    if xmax != 0:
        return np.linalg.norm(x / xmax) * xmax
    else:
        return np.linalg.norm(x)

def load_ratings(data_file=DATA_FILE):
    data_file = DATA_FILE
    ratings = []
    with open(data_file, 'r') as file:
        # Read header.
        _ = file.readline()
        for line in file:
            key, value_string = line.split(",")
            rating = int(value_string)
            row_string, col_string = key.split("_")
            row = int(row_string[1:])
            col = int(col_string[1:])
            ratings.append((row - 1, col - 1, rating))
    return ratings

def mask_validation(data):
    masked_data = np.copy(data)
    validation_indices = get_indeces_from_file(VALIDATION_FILE_NAME)
    for row_index, col_index in validation_indices:
        masked_data[row_index][col_index] = 0
    return masked_data

def ratings_to_matrix(ratings):
    """Converts a list of ratings to a numpy matrix."""
    matrix_rows = USER_COUNT
    matrix_cols = ITEM_COUNT
    print("Building [%d x %d] rating matrix." % (matrix_rows, matrix_cols))
    matrix = np.zeros([matrix_rows, matrix_cols])
    for (row, col, rating) in ratings:
        matrix[row, col] = rating
    print("Finished building rating matrix.")
    return matrix

def impute(data, reconstruction):
    observed_indeces = get_observed_indeces(data)
    for row_index, col_index in observed_indeces:
        reconstruction[row_index][col_index] = data[row_index][col_index]
    return reconstruction

def get_validation_indices():
    validation_indices = get_indeces_from_file(VALIDATION_FILE_NAME)
    return validation_indices

def get_observed_indeces(data):
    row_indices, col_indices = np.where(data != 0)
    return list(zip(row_indices, col_indices))

def get_unobserved_indeces(data):
    row_indices, col_indices = np.where(data == 0)
    return list(zip(row_indices, col_indices))

def get_indeces_from_file(file_name):
    indeces = []
    with open(file_name, 'r') as file:
        # Read header.
        _ = file.readline()
        for line in file:
            i, j = line.split(",")
            indeces.append((int(i), int(j)))
    return indeces

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

def reconstruction_to_predictions(reconstruction, submission_file):
    indices_to_predict = get_indices_to_predict()
    predictions = list(map(lambda t: \
            (t[0] + 1, t[1] + 1, reconstruction[t[0], t[1]]), \
            indices_to_predict))
    write_ratings(predictions, submission_file)

def clip(data):
    data[data > 5] = 5
    data[data < 1] = 1
    return data

def predict_by_avg(data, by_row):
    data = data.T if by_row else data
    for row in data:
        empty = (row == 0)
        row_sum = np.sum(row)
        row[empty] = row_sum / np.count_nonzero(row)
    return data.T if by_row else data

def novel_init(data):
    global_average = np.sum(data) / np.count_nonzero(data)
    global_variance = np.var(data[data != 0])

    m = np.zeros((data.shape[1],))
    for i in range(data.shape[1]):
        ratings = data[:, i]
        ratings = ratings[ratings != 0]
        movie_variance = np.var(ratings)
        k = movie_variance / global_variance
        m[i] = (global_average * k + np.sum(ratings)) /\
               (k + np.count_nonzero(ratings))

    u = np.zeros((data.shape[0],))

    for i in range(data.shape[0]):
        user_ratings = data[i]
        diff = m - user_ratings
        r = diff[user_ratings != 0]
        k = np.var(r) / global_variance
        u[i] = (global_average * k + sum(r)) / (k + sum(r))

    w = 10.0
    user_counts = np.count_nonzero(data, axis=1)
    movie_counts = np.count_nonzero(data, axis=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d = movie_counts[j] / (movie_counts[j] + w * user_counts[i])
            data[i, j] = d * m[j] + (1 - d) * u[i]

    return data

def predict_bias(data):
    total_average = np.mean(data[np.nonzero(data)])
    row_biases = np.zeros(data.shape[0])
    col_biases = np.zeros(data.shape[1])

    for row_index in range(data.shape[0]):
        row_biases[row_index] = np.sum(data[row_index]) / \
                np.count_nonzero(data[row_index]) - total_average

    # plt.hist(row_biases)
    # plt.show()

    for col_index in range(data.shape[1]):
        col_biases[col_index] = np.sum(data[:][col_index]) / \
                                np.count_nonzero(data[:][col_index]) - total_average

    # plt.hist(col_biases)
    # plt.show()

    counter = 0
    values = np.zeros(10000000)

    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            if data[row_index, col_index] == 0:
                new_value = total_average + \
                        row_biases[row_index] + col_biases[col_index]
                data[row_index, col_index] = new_value
                values[counter] = new_value
                counter += 1
    # plt.hist(values)
    # plt.show()
    return data

def compute_rsme(data, prediction):
    validation_indices = get_indeces_from_file(VALIDATION_FILE_NAME)
    squared_error = 0
    for i, j in validation_indices:
        squared_error += (data[i][j] - prediction[i][j]) ** 2
    return np.sqrt(squared_error / len(validation_indices))

def knn_smoothing(data, user_embeddings):
    normalized_user_embeddings = sklearn.preprocessing.normalize(user_embeddings)
    n_neighbors = 3
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(normalized_user_embeddings)
    distances, neighbors = knn.kneighbors(normalized_user_embeddings)
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]

    ones = np.ones(distances.shape)
    similarities = ones - distances
    weights = np.square(np.square(similarities))
    smoothed_data = np.zeros(data.shape)
    aggregated_neighbor_ratings = np.zeros(data.shape)

    for i in range(data.shape[0]):
        stacked_ratings = []
        for neighbor in neighbors[i]:
            stacked_ratings.append(data[neighbor])
        stacked_ratings = np.asarray(stacked_ratings)
        aggregated_neighbor_ratings[i] = np.matmul(weights[i], stacked_ratings) / sum(weights[i])

    weight_knn = 0.01
    for i in range(data.shape[0]):
        smoothed_data[i] = (1 - weight_knn) * data[i] + weight_knn * aggregated_neighbor_ratings[i]

    smoothed_data = clip(smoothed_data)
    return smoothed_data





