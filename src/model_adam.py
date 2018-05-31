from random import shuffle
import os
import sys
import numpy as np
import utils
import utils_sgd

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_adam.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/adam_scores.csv')
N_EPOCHS = 1
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.0001

BETA_1 = 0.9
BETA_2 = 0.999
ALPHA = 0.001
E = 1e-8

def learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        regularization):
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(data[np.nonzero(data)])
    approximation_rank = u_embedding.shape[1]

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

    for i in range(n_epochs):
        print("Epoch {0}:".format(i))
        shuffle(training_indices)
        for k, l in training_indices:

            c = np.random.randint(1, 5)

            residual = data[k, l] - total_average - u_bias[k] - z_bias[l]\
                    - np.dot(u_embedding[k, :], z_embedding[l, :])

            if c == 1:
                u_counters[k] += 1
                u_update = -residual * z_embedding[l, :] + \
                        utils.safe_norm(regularization * \
                        u_embedding[k, :])
                m_u[k] = BETA_1 * m_u[k] + (1 - BETA_1) * u_update
                v_u[k] = BETA_2 * v_u[k] + (1 - BETA_2) * np.square(u_update)
                m = m_u[k] / (1 - BETA_1**u_counters[k])
                v = v_u[k] / (1 - BETA_2**u_counters[k])
                u_embedding[k, :] += np.divide(LEARNING_RATE * m, np.sqrt(v) + E)
            elif c == 2:

                z_counters[l] += 1
                z_update = - residual * u_embedding[k, :] + \
                        utils.safe_norm(regularization * \
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

def predict_by_adam(data, approximation_rank=None,
        regularization=REGULARIZATION, n_epochs=N_EPOCHS, u_embedding=None,
        z_embedding=None):
    np.random.seed(21)
    if u_embedding is None and z_embedding is None:
        print("Initialize embeddings.")
        u_embedding, z_embedding = utils_sgd.get_initialized_embeddings(
                approximation_rank, data.shape[0], data.shape[1])
    if u_embedding is None or z_embedding is None:
        raise ValueError("embedding is None!")
    u_bias, z_bias = utils_sgd.get_initialized_biases(data)
    learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        regularization)
    total_average = np.mean(data[np.nonzero(data)])
    reconstruction = utils_sgd.reconstruct(u_embedding, z_embedding, total_average, u_bias, z_bias)
    utils.clip(reconstruction)
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
    data = utils.mask_validation(data)
    reconstruction = predict_by_adam(data, k, regularization)
    rsme = utils.compute_rsme(data, reconstruction)
    utils_sgd.write_sgd_score(rsme, k, regularization, SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
