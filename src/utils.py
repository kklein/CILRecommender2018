import os
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_FILE = os.path.join(ROOT_DIR, 'data/data_train.csv')
TRAINING_FILE_NAME = os.path.join(ROOT_DIR,\
        'data/trainingIndices.csv')
VALIDATION_FILE_NAME = os.path.join(ROOT_DIR,\
        'data/validationIndices.csv')

SAMPLE_SUBMISSION = os.path.join(ROOT_DIR,\
        'data/sampleSubmission.csv')
ITEM_COUNT = 1000
USER_COUNT = 10000

def load_ratings():
    """Loads the rating data from the specified file.
    Does not yet build the rating matrix. Use 'ratings_to_matrix' to do that.
    Assumes the file has a header (which is ignored), and that the ratings are
    then specified as 'rXXX_cXXX,X', where the 'X' blanks specify the row, the
    column, and then the actual (integer) rating.
    """
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

            if rating < 1 or rating > 5:
                raise ValueError("Found illegal rating value [%d]." % rating)

            ratings.append((row - 1, col - 1, rating))
    return ratings

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

def predict_bias(data):
    total_average = np.mean(data[np.nonzero(data)])
    row_biases = np.zeros(data.shape[0])
    col_biases = np.zeros(data.shape[1])

    for row_index in range(data.shape[0]):
        row_biases[row_index] = np.sum(data[row_index]) / \
                np.count_nonzero(data[row_index]) - total_average

    plt.hist(row_biases)
    plt.show()

    for col_index in range(data.shape[1]):
        col_biases[col_index] = np.sum(data[:][col_index]) / \
                np.count_nonzero(data[:][col_index]) - total_average

    plt.hist(col_biases)
    plt.show()

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
    plt.hist(values)
    plt.show()

    print('filled %d many holes' % counter)
    return data

def compute_rsme(data, prediction):
    validation_indices = get_indeces_from_file(VALIDATION_FILE_NAME)
    squared_error = 0
    for i, j in validation_indices:
        squared_error += (data[i][j] - prediction[i][j]) ** 2
    return np.sqrt(squared_error / len(validation_indices))
