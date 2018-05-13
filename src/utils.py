import os
import numpy as np

#TODO(kkleindev): Find execution-folder-independent approach.
ROOT_DIR = os.path.dirname(os.path.abspath(''))
TRAINING_FILE_NAME = os.path.join(ROOT_DIR,\
        'data/trainingIndices.csv')
VALIDATION_FILE_NAME = os.path.join(ROOT_DIR,\
        'data/validationIndices.csv')
ITEM_COUNT = 1000
USER_COUNT = 10000

def load_ratings(data_file):
    """Loads the rating data from the specified file.
    Does not yet build the rating matrix. Use 'ratings_to_matrix' to do that.
    Assumes the file has a header (which is ignored), and that the ratings are
    then specified as 'rXXX_cXXX,X', where the 'X' blanks specify the row, the
    column, and then the actual (integer) rating.
    """
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

def compute_rsme(data, prediction):
    validation_indices = get_indeces_from_file(VALIDATION_FILE_NAME)
    squared_error = 0
    for i, j in validation_indices:
        squared_error += (data[i][j] - prediction[i][j]) ** 2
    return np.sqrt(squared_error / len(validation_indices))
