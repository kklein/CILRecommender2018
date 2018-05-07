import sys
import time
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import copy


# Rows are users
USER_COUNT = 10000
# Columns are items
ITEM_COUNT = 1000
SUBMISSION_FILE = '../data/submission_nn.csv'
SAMPLE_SUBMISSION = '../data/sampleSubmission.csv'
N_EPOCHS = 20
LEARNING_RATE = 0.01
EPSILON = 0.0001

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

def get_indices_to_predict():
    """Get list of indices to predict from sample submission file.
        Returns:
            indices_to_predict:  list of tuples with indices"""

    indices_to_predict = []
    with open(SAMPLE_SUBMISSION, 'r') as file:
        header = file.readline()
        for line in file:
            key, value_string = line.split(",")
            row_string, col_string = key.split("_")
            i = int(row_string[1:]) - 1
            j = int(col_string[1:]) - 1
            indices_to_predict.append((i, j))

    return indices_to_predict

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
    u = np.random.rand(data.shape[0], approximation_rank) 
    z = np.random.rand(data.shape[1], approximation_rank) 

    n_samples = int(0.05 * len(observed_indices))

    prev_loss = sys.float_info.max
    for i in range(N_EPOCHS):
        print("Epoch {0}:".format(i))

        for j in range(n_samples):
            index = np.random.randint(0, len(observed_indices) - 1)
            k, l = observed_indices[index]
            u[k, :] += LEARNING_RATE * (data[k, l] - np.dot(u[k, :], z[l, :])) \
                    * z[l, :]

        for j in range(n_samples):
            index = np.random.randint(0, len(observed_indices) - 1)
            k, l = observed_indices[index]
            z[l, :] += LEARNING_RATE * (data[k, l] - np.dot(u[k, :], z[l, :])) \
                    * u[k, :]

        prod = np.matmul(u, z.T)
        prod[data == 0] = 0
        diff = data - prod
        square = np.multiply(diff, diff)
        loss = np.sum(square)
        print("Loss {0}".format(loss))
        print("Loss ratio {0}: ".format((prev_loss - loss) / loss))
        if (prev_loss - loss) / loss < EPSILON:
            break
        prev_loss = loss
    return np.dot(u, z.T)


def get_embeddings_by_svd(data):
    """Given data matrix, returns user embeddings and item embeddings."""
    u, s, vh = np.linalg.svd(data)
    return u, vh.T


def prepare_data_for_nn(user_embeddings, item_embeddings, data_matrix):
    """Concatenates user embeddings and item embeddings, and adds corresponding rating from data matrix.
    Returns: x_train, y_train, x_validate, y_validate."""

    x_train = []
    y_train = []

    x_validate = []
    y_validate = []

    counter = 0
    for i, j in zip(*np.nonzero(data_matrix)):
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        y = data_matrix[i, j]
        x_train.append(x)
        y_train.append(y)
        counter += 1
        if counter > 1000:
            break
        elif counter % 1000 == 0:
            print(counter)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_train = np.ravel(y_train)

    indices_to_predict = get_indices_to_predict()
    counter = 0

    # TODO: Remove y_validate, these are the values we want to predict!
    for i, j in indices_to_predict:
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        y = data_matrix[i, j]
        x_validate.append(x)
        y_validate.append(y)
        counter += 1
        if False:
            break
        elif counter % 1000 == 0:
            print(counter)

    x_validate = np.asarray(x_validate)
    y_validate = np.asarray(y_validate)
    y_validate = np.ravel(y_validate)

    return x_train, y_train, x_validate, y_validate

def write_nn_predictions(data_matrix, y_predicted):

    indices_to_predict = get_indices_to_predict()
    for a, index in enumerate(indices_to_predict):
        data_matrix[index] = y_predicted[a]

    reconstruction_to_predictions(data_matrix)



def predict_by_nn(data_matrix, imputed_data):

    # Get embeddings
    user_embeddings, item_embeddings = get_embeddings_by_svd(imputed_data)

    x_train, y_train, x_validate, y_validate = prepare_data_for_nn(user_embeddings, item_embeddings, data_matrix)
    print(y_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

    classifier = MLPClassifier((50, 50))

    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)
    print(accuracy)

    y_predicted = classifier.predict(x_validate)
    write_nn_predictions(data_matrix, y_predicted)



def main():
    all_ratings = load_ratings('../data/data_train.csv')
    data_matrix = ratings_to_matrix(all_ratings, USER_COUNT, ITEM_COUNT)
    #test_predict_by_avg()
    imputed_data = predict_by_avg(copy.copy(data_matrix), True)
    #reconstruction = predict_by_svd(imputed_data, 2)
    #reconstruction = predict_by_sgd(data_matrix, 10)
    #reconstruction_to_predictions(reconstruction)

    predict_by_nn(data_matrix, imputed_data)




if __name__ == '__main__':
    main()
