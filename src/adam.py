from random import shuffle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_adam.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/adam_scores.csv')
N_EPOCHS = 15
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.0001

BETA_1 = 0.9

BETA_2 = 0.999
ALPHA = 0.001
E = 1e-8

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
    np.random.seed(21)
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    u_embedding = np.random.rand(data.shape[0], approximation_rank).astype(
            np.float128)
    z_embedding = np.random.rand(data.shape[1], approximation_rank).astype(
            np.float128)
    u_bias, z_bias = get_initialized_biases(data, training_indices)
    total_average = np.mean(data[np.nonzero(data)])

    u_counters = np.zeros(data.shape[0])
    z_counters = np.zeros(data.shape[1])
    u_bias_counters = np.zeros(data.shape[0])
    z_bias_counters = np.zeros(data.shape[0])
    m_u = np.zeros((data.shape[0], approximation_rank))
    v_u = np.zeros((data.shape[0], approximation_rank))
    m_z = np.zeros((data.shape[1], approximation_rank))
    v_z = np.zeros((data.shape[1], approximation_rank))
    m_u_bias = np.zeros(data.shape[0])
    v_u_bias = np.zeros(data.shape[0])
    m_z_bias = np.zeros(data.shape[1])
    v_z_bias = np.zeros(data.shape[1])

    # rsmes = []
    for i in range(N_EPOCHS):
        print("Epoch {0}:".format(i))
        shuffle(training_indices)
        for k, l in training_indices:

            c = np.random.randint(1,5)

            residual = data[k, l] - total_average - u_bias[k] - z_bias[l]\
                    - np.dot(u_embedding[k, :], z_embedding[l, :])

            if c == 1:
                u_counters[k] += 1
                u_update = -residual * z_embedding[l, :] + \
                        safe_norm(regularization * \
                        u_embedding[k, :])
                m_u[k] = BETA_1 * m_u[k] + (1 - BETA_1) * u_update
                v_u[k] = BETA_2 * v_u[k] + (1 - BETA_2) * np.square(u_update)
                m = m_u[k] / (1 - BETA_1**u_counters[k])
                v = v_u[k] / (1 - BETA_2**u_counters[k])
                u_embedding[k, :] += np.divide(LEARNING_RATE * m, np.sqrt(v) + E)
            elif c == 2:

                z_counters[l] += 1
                z_update = - residual * u_embedding[k, :] + \
                        safe_norm(regularization * \
                        z_embedding[l, :])
                m_z[l] = BETA_1 * m_z[l] + (1 - BETA_1) * z_update
                v_z[l] = BETA_2 * v_z[l] + (1 - BETA_2) * np.square(z_update)
                m = m_z[l] / (1 - BETA_1**z_counters[l])
                v = v_z[l] / (1 - BETA_2**z_counters[l])
                z_embedding[l, :] += np.divide(LEARNING_RATE * m, np.sqrt(v) + E)

            elif c == 3:
                u_bias_counters[k] += 1
                u_bias_update = - residual + regularization * np.absolute(u_bias[k])
                m_u_bias[k] = BETA_1 * m_u_bias[k] + (1 - BETA_1) * u_bias_update
                v_u_bias[k] = BETA_2 * v_u_bias[k] + (1 - BETA_2) * np.square(u_bias_update)
                m = m_u_bias[k] / (1 - BETA_1**u_bias_counters[k])
                v = v_u_bias[l] / (1 - BETA_2**u_bias_counters[k])
                u_bias[k] += np.divide(LEARNING_RATE * m, np.sqrt(v) + E)

            elif c == 4:
                z_bias_counters[l] += 1
                z_bias_update = - residual + regularization * np.absolute(z_bias[l])
                m_z_bias[l] = BETA_1 * m_z_bias[l] + (1 - BETA_1) * z_bias_update
                v_z_bias[l] = BETA_2 * v_z_bias[l] + (1 - BETA_2) * np.square(z_bias_update)
                m = m_z_bias[l] / (1 - BETA_1**z_bias_counters[l])
                v = v_z_bias[l] / (1 - BETA_2**z_bias_counters[l])
                z_bias[l] += np.divide(LEARNING_RATE * m, np.sqrt(v) + E)

        reconstruction = reconstruct(u_embedding, z_embedding, total_average,
                u_bias, z_bias)
        rsme = utils.compute_rsme(data, reconstruction)
        # rsmes.append((i, rsme))
        print('RSME: %f' % rsme)

    # x, y = zip(*rsmes)
    # plt.scatter(x, y)
    # plt.show()
    write_sgd_score(rsme, approximation_rank, regularization)
    return reconstruction

def main():
    # k = 10
    k = int(sys.argv[1])
    # regularization = REGULARIZATION
    regularization = float(sys.argv[2])
    # ranks = [i for i in range(3, 35)]
    # regularizations = [0.0005 * i for i in range(40)]
    # k = np.random.choice(ranks)
    # regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    reconstruction = predict_by_sgd(data, k, regularization)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
