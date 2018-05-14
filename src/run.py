import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import utils


DATA_FILE = os.path.join(utils.ROOT_DIR, 'data/data_train.csv')
SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sgd.csv')
SAMPLE_SUBMISSION = os.path.join(utils.ROOT_DIR,\
        'data/sampleSubmission.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/biased_sgd_scores.csv')
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
        # Read header.
        _ = file.readline()
        for line in file:
            key, _ = line.split(",")
            row_string, col_string = key.split("_")
            i = int(row_string[1:])
            j = int(col_string[1:])
            predictions.append((i, j, reconstruction[i-1, j-1]))
    write_ratings(predictions)

def write_sgd_score(score, k, regularization):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %f, %f\n' % (k, regularization, score))

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

def predict_by_svd(data, approximation_rank):
    imputed_data = predict_by_avg(data, True)
    # imputed_data = predict_bias(data)
    u_embeddings, singular_values, vh_embeddings =\
            np.linalg.svd(imputed_data)
    u_embeddings = u_embeddings[:, 0:approximation_rank]
    singular_values = singular_values[0:approximation_rank]
    vh_embeddings = vh_embeddings[0:approximation_rank, :]
    return np.dot(u_embeddings,\
            np.dot(np.diag(singular_values), vh_embeddings))

def clip(data):
    data[data > 5] = 5
    data[data < 1] = 1
    return data

def predict_by_sgd(data, approximation_rank, regularization):
    observed_indices = utils.get_observed_indeces(data)
    total_average = np.mean(data[np.nonzero(data)])
    u_embedding = np.random.rand(data.shape[0], approximation_rank)
    z_embedding = np.random.rand(data.shape[1], approximation_rank)

    u_bias = np.zeros(data.shape[0])
    u_counters = np.zeros(data.shape[0])
    z_bias = np.zeros(data.shape[1])
    z_counters = np.zeros(data.shape[1])
    for k, l in observed_indices:
        u_bias[k] += data[k][l]
        u_counters[k] += 1
        z_bias[l] += data[k][l]
        z_counters[l] += 1
    for k in range(data.shape[0]):
        u_bias[k] = (u_bias[k] / u_counters[k]) - total_average
    for l in range(data.shape[1]):
        z_bias[l] = (z_bias[l] / z_counters[l]) - total_average

    n_samples = int(0.2 * len(observed_indices))
    prev_loss = sys.float_info.max
    # rsmes = []
    for i in range(N_EPOCHS):
        print("Epoch {0}:".format(i))

        for _ in range(n_samples):
            index = np.random.randint(0, len(observed_indices) - 1)
            k, l = observed_indices[index]
            residual = data[k, l] - total_average - u_bias[k] - z_bias[l]\
                    - np.dot(u_embedding[k, :], z_embedding[l, :])
            u_update = LEARNING_RATE * (residual * z_embedding[l, :] - \
                    regularization * np.linalg.norm(u_embedding[k, :]))
            z_update = LEARNING_RATE * (residual * u_embedding[k, :] - \
                    regularization * np.linalg.norm(z_embedding[l, :]))
            u_bias_update = LEARNING_RATE * (residual - regularization * \
                    u_bias[k])
            z_bias_update = LEARNING_RATE * (residual - regularization * \
                    z_bias[l])
            u_embedding[k, :] += u_update
            z_embedding[l, :] += z_update
            u_bias[k] += u_bias_update
            z_bias[l] += z_bias_update

        # prod = np.matmul(u_embedding, z_embedding.T)
        # prod[data == 0] = 0
        # diff = data - prod
        # square = np.multiply(diff, diff)
        # loss = np.sum(square)
        # print("Loss: {0}".format(loss))
        # print("Loss ratio {0}: ".format((prev_loss - loss) / loss))
        # if (prev_loss - loss) / loss < EPSILON:
        #     break
        # prev_loss = loss
        # rsmes.append((i, utils.compute_rsme(data, prod)))
    # x, y = zip(*rsmes)
    # plt.plot(x, y)
    # plt.show()
    reconstruction = np.dot(u_embedding, z_embedding.T) + total_average
    # TODO(kkkleindev): Replace this by appropriate broadcasting.
    for l in range(data.shape[1]):
        reconstruction[:, l] += u_bias
    for k in range(data.shape[0]):
        reconstruction[k, :] += z_bias
    rsme = utils.compute_rsme(data, reconstruction)
    write_sgd_score(rsme, approximation_rank, regularization)
    return reconstruction

def main():
    # k = 10
    k = int(sys.argv[1])
    # regularization = REGULARIZATION
    regularization = float(sys.argv[2])
    # ranks = [5 * i for i in range(1, 40)]
    # regularizations = [0.0005 * i for i in range(10)]
    # k = np.random.choice(ranks)
    # regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings(DATA_FILE)
    data = utils.ratings_to_matrix(all_ratings)
    # reconstruction = predict_by_svd(data, 10)
    reconstruction = predict_by_sgd(data, k, regularization)
    reconstruction = clip(reconstruction)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME: %f' % rsme)
    reconstruction_to_predictions(reconstruction)

if __name__ == '__main__':
    main()
