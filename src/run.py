import sys
import numpy as np
import matplotlib.pyplot as plt
import utils

SUBMISSION_FILE = '../data/submission_sgd.csv'
SAMPLE_SUBMISSION = '../data/sampleSubmission.csv'
SCORE_FILE = '../data/sgd_scores.csv'
N_EPOCHS = 150
LEARNING_RATE = 0.001
REGULARIZATION = 0.000
EPSILON = 0.0001

def write_ratings(predictions):
    with open(SUBMISSION_FILE, 'w') as file:
        file.write('Id,Prediction\n')
        for i, j, prediction in predictions:
            file.write('r%d_c%d,%f\n' % (i, j, prediction))

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

def write_sgd_score(score, k, Lambda):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %f, %f\n' % (k, Lambda, score))

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

def clip(data):
    data[data > 5] = 5
    data[data < 1] = 1
    return data

def predict_by_sgd(data, approximation_rank, Lambda):
    observed_indices = utils.get_observed_indeces(data)
    u = np.random.rand(data.shape[0], approximation_rank)
    z = np.random.rand(data.shape[1], approximation_rank)
    n_samples = int(0.2 * len(observed_indices))
    prev_loss = sys.float_info.max
    # rsmes = []
    for i in range(N_EPOCHS):
        print("Epoch {0}:".format(i))

        for sample_index in range(n_samples):
            index = np.random.randint(0, len(observed_indices) - 1)
            k, l = observed_indices[index]
            residual = data[k, l] - np.dot(u[k, :], z[l, :])
            u_update = LEARNING_RATE * (residual * z[l, :] - \
                    # REGULARIZATION * np.linalg.norm(u[k, :]))
                    Lambda * np.linalg.norm(u[k, :]))

            z_update = LEARNING_RATE * (residual * u[k, :] - \
                    # REGULARIZATION * np.linalg.norm(z[l, :]))
                    Lambda * np.linalg.norm(z[l, :]))

            u[k, :] += u_update
            z[l, :] += z_update

        prod = np.matmul(u, z.T)
        prod[data == 0] = 0
        diff = data - prod
        square = np.multiply(diff, diff)
        loss = np.sum(square)
        print("Loss: {0}".format(loss))
        print("Loss ratio {0}: ".format((prev_loss - loss) / loss))
        if (prev_loss - loss) / loss < EPSILON:
            break
        prev_loss = loss
        # rsmes.append((i, utils.compute_rsme(data, prod)))
    # x, y = zip(*rsmes)
    # plt.plot(x, y)
    # plt.show()
    reconstruction = np.dot(u, z.T)
    #rsme = utils.compute_rsme(data, reconstruction)
    #write_sgd_score(rsme, approximation_rank, Lambda)
    return reconstruction

def main():
    k = int(sys.argv[1])
    Lambda = float(sys.argv[2])
    all_ratings = utils.load_ratings('../data/data_train.csv')
    data_matrix = utils.ratings_to_matrix(all_ratings)
    #test_predict_by_avg()
    #imputed_data = predict_by_avg(data_matrix, True)
    #imputed_data = predict_bias(data_matrix)
    #reconstruction = predict_by_svd(imputed_data, 10)
    reconstruction = predict_by_sgd(data_matrix, k, Lambda)
    reconstruction = clip(reconstruction)
    rsme = utils.compute_rsme(data_matrix, reconstruction)
    print('RSME: %f' % rsme)
    reconstruction_to_predictions(reconstruction)

if __name__ == '__main__':
    main()
