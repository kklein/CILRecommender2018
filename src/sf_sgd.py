from random import shuffle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sf_sgd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/sf_sgd_scores.csv')
N_EPOCHS = 2
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.0001

# https://stackoverflow.com/questions/42746248/numpy-linalg-norm-behaving-oddly-wrongly
def safe_norm(x):
    xmax = np.max(x)
    if xmax != 0:
        return np.linalg.norm(x / xmax) * xmax
    else:
        return np.linalg.norm(x)

def write_sgd_score(score, k, regularization):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %f, %f\n' % (k, regularization, score))

def get_initialized_biases(data, training_indices):
    u_bias = np.zeros(data.shape[0], dtype=np.float128)
    u_counters = np.zeros(data.shape[0], dtype=np.float128)
    z_bias = np.zeros(data.shape[1], dtype=np.float128)
    z_counters = np.zeros(data.shape[1], dtype=np.float128)
    total_average = np.mean(data[np.nonzero(data)])
    for k, l in training_indices:
        u_bias[k] += data[k][l]
        u_counters[k] += 1
        z_bias[l] += data[k][l]
        z_counters[l] += 1
    for k in range(data.shape[0]):
        u_bias[k] = ((u_bias[k] + 25 * total_average) / (25 + u_counters[k]))\
                - total_average
        # u_bias[k] = (u_bias[k] / u_counters[k]) - total_average
    for l in range(data.shape[1]):
        # z_bias[l] = (z_bias[l] / z_counters[l]) - total_average
        z_bias[l] = ((z_bias[l] + 25 * total_average) / (25 + z_counters[l]))\
                - total_average
    return u_bias, z_bias

def reconstruct(u_embedding, z_embedding, total_average, u_bias, z_bias):

    # Reshape arrays in order to allow for broadcasting on matrix.
    u_bias = np.reshape(u_bias, (u_bias.shape[0], 1))
    z_bias = np.reshape(z_bias, (1, z_bias.shape[0]))

    prod = np.dot(u_embedding, z_embedding.T)
    prod += total_average
    prod += u_bias
    prod += z_bias
    return utils.clip(prod)

def predict_by_sgd(data, approximation_rank, regularization):
    np.random.seed(42)
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    u_embedding = np.random.rand(data.shape[0], approximation_rank).astype(
            np.float128)
    z_embedding = np.random.rand(data.shape[1], approximation_rank).astype(
            np.float128)
    u_bias, z_bias = get_initialized_biases(data, training_indices)
    total_average = np.mean(data[np.nonzero(data)])

    for feature_index in range(approximation_rank):
        print("Feature %d." % feature_index)
        # rsmes = []
        for i in range(N_EPOCHS):
            print("Epoch {0}:".format(i))
            shuffle(training_indices)
            for k, l in training_indices:
                residual = data[k, l] - total_average - u_bias[k] - z_bias[l]\
                        - np.dot(u_embedding[k, :feature_index + 1], z_embedding[l, :feature_index + 1])
                u_update = LEARNING_RATE * residual * z_embedding[l, feature_index] - \
                        safe_norm(LEARNING_RATE * regularization * \
                        u_embedding[k, feature_index])
                z_update = LEARNING_RATE * residual * u_embedding[k, feature_index] - \
                        safe_norm(LEARNING_RATE * regularization * \
                        z_embedding[l, feature_index])
                u_bias_update = LEARNING_RATE * (residual - regularization *
                        np.absolute(u_bias[k]))
                z_bias_update = LEARNING_RATE * (residual - regularization *
                        np.absolute(z_bias[l]))
                u_embedding[k, feature_index] += u_update
                z_embedding[l, feature_index] += z_update
                u_bias[k] += u_bias_update
                z_bias[l] += z_bias_update
            # reconstruction = reconstruct(u_embedding[:, :feature_index + 1], z_embedding[:, :feature_index + 1], total_average,
                    # u_bias, z_bias)
            # rsme = utils.compute_rsme(data, reconstruction)
            # rsmes.append((i, rsme))
            # print('RSME: %f' % rsme)

    # x, y = zip(*rsmes)
    # plt.scatter(x, y)
    # plt.show()
    return reconstruction

def main():
    # k = 10
    # k = int(sys.argv[1])
    # regularization = REGULARIZATION
    # regularization = float(sys.argv[2])
    ranks = [i for i in range(3, 100)]
    regularizations = [0.0005 * i for i in range(400)]
    k = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    reconstruction = predict_by_sgd(masked_data, k, regularization)
    rsme = utils.compute_rsme(data, reconstruction)
    write_sgd_score(rsme, approximation_rank, regularization)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
