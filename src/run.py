import sys
import numpy as np

# Rows are users
USER_COUNT = 10000
# Columns are items
ITEM_COUNT = 1000
SUBMISSION_FILE = '../data/submission_sgd.csv'
SAMPLE_SUBMISSION = '../data/sampleSubmission.csv'
N_EPOCHS = 5
LEARNING_RATE = 0.05

def load_ratings(data_file):
    """Loads the rating data from the specified file.
    Does not yet build the rating matrix. Use 'ratings_to_matrix' to do that.
    Assumes the file has a header (which is ignored), and that the ratings are
    then specified as 'rXXX_cXXX,X', where the 'X' blanks specify the row, the
    column, and then the actual (integer) rating.
    """
    ratings = []
    with open(data_file, 'r') as file:
        header = file.readline()
        # print("Header: %s" % header)
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

def ratings_to_matrix(ratings, matrix_rows, matrix_cols):
    """Converts a list of ratings to a numpy matrix."""
    print("Building [%d x %d] rating matrix." % (matrix_rows, matrix_cols))
    matrix = np.zeros([matrix_rows, matrix_cols])
    for (row, col, rating) in ratings:
        matrix[row, col] = rating
    print("Finished building rating matrix.")
    return matrix

def reconstruction_to_predictions(reconstruction):
    predictions = []
    with open(SAMPLE_SUBMISSION, 'r') as file:
        header = file.readline()
        for line in file:
            key, value_string = line.split(",")
            row_string, col_string = key.split("_")
            i = int(row_string[1:])
            j = int(col_string[1:])
            predictions.append((i, j, reconstruction[i-1, j-1]))
    write_ratings(predictions)

def write_ratings(predictions):
    with open(SUBMISSION_FILE, 'w') as file:
        file.write('Id,Prediction\n')
        for i, j, prediction in predictions:
            file.write('r%d_c%d,%f\n' % (i, j, prediction))

def predict_by_avg(data, by_row):
    data = data.T if by_row else data
    for row in data:
        empty = (row == 0)
        row_sum = np.sum(row)
        row[empty] = row_sum / np.count_nonzero(row)
    return data.T if by_row else data

def test_predict_by_avg():
    a = np.array([[1, 0, 2], [0, 2, 3], [0, 0, 1]])
    expected_array = np.array([[1, 2, 2], [1, 2, 3], [1, 2, 1]])
    actual_array = predict_by_avg(a, True)
    assert np.array_equal(actual_array, expected_array)

def predict_by_svd(data, approximation_rank):
    u, s, vh = np.linalg.svd(data)
    u = u[:, 0:approximation_rank]
    s = s[0:approximation_rank]
    vh = vh[0:approximation_rank, :]
    return np.dot(u, np.dot(np.diag(s), vh))

def predict_by_sgd(data, approximation_rank):
    row_indices, col_indices = np.where(data != 0)
    observed_indices = list(zip(row_indices, col_indices))
    u = np.random.rand(data.shape[0], approximation_rank) * 4 + 1
    z = np.random.rand(data.shape[1], approximation_rank) * 4 + 1

    n_samples = int(0.001 * len(observed_indices))

    prev_loss = sys.maxsize
    for i in range(N_EPOCHS):
        print("Epoch {0}:".format(i))
        for j in range(n_samples):
            index = np.random.choice(range(len(observed_indices)))
            k, l = observed_indices[index]
            u[k, :] -= LEARNING_RATE * (data[k, l] - np.dot(u[k, :], z[l, :])) \
                    * z[l, :]
        for j in range(n_samples):
            index = np.random.choice(range(len(observed_indices)))
            k, l = observed_indices[index]
            z[l, :] -= LEARNING_RATE * (data[k, l] - np.dot(u[k, :], z[l, :])) \
                    * u[k, :]
        prod = np.matmul(u, z.T)
        prod[data == 0] = 0
        diff = data - prod
        square = np.multiply(diff, diff)
        loss = np.matrix.sum(square)
        print("Loss {0}".format(loss))
        if (prev_loss - loss) / loss < epsilon:
            break
    return np.dot(u, z.T)



def main():
    all_ratings = load_ratings('../data/data_train.csv')
    data_matrix = ratings_to_matrix(all_ratings, USER_COUNT, ITEM_COUNT)
    #test_predict_by_avg()
    #imputed_data = predict_by_avg(data_matrix, True)
    #reconstruction = predict_by_svd(imputed_data, 2)
    reconstruction = predict_by_sgd(data_matrix, 10)
    reconstruction_to_predictions(reconstruction)

if __name__ == '__main__':
    main()
