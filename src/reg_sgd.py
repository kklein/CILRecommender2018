from random import shuffle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sgd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/biased_sgd_scores.csv')
N_EPOCHS = 25
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.0001

def write_sgd_score(score, k, regularization):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %f, %f\n' % (k, regularization, score))

def predict_by_sgd(data, approximation_rank, regularization):
    np.random.seed(42)
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(data[np.nonzero(data)])
    u_embedding = np.random.rand(data.shape[0], approximation_rank).astype(
            np.float128)
    z_embedding = np.random.rand(data.shape[1], approximation_rank).astype(
            np.float128)

    u_bias = np.zeros(data.shape[0], dtype=np.float128)
    u_counters = np.zeros(data.shape[0], dtype=np.float128)
    z_bias = np.zeros(data.shape[1], dtype=np.float128)
    z_counters = np.zeros(data.shape[1], dtype=np.float128)
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

    n_samples = int(0.2 * len(training_indices))
    prev_loss = sys.float_info.max
    # rsmes = []
    for i in range(N_EPOCHS):
        print("Epoch {0}:".format(i))

        # for _ in range(n_samples):
        shuffle(training_indices)
        for k, l in training_indices:
            # index = np.random.randint(0, len(training_indices) - 1)
            # k, l = training_indices[index]
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
    # = int(sys.argv[1])
    # regularization = REGULARIZATION
    #regularization = float(sys.argv[2])
    ranks = [5 * i for i in range(1, 30)]
    regularizations = [0.004 * i for i in range(100)]
    k = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    reconstruction = predict_by_sgd(data, k, regularization)
    reconstruction = utils.clip(reconstruction)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME: %f' % rsme)
    # utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
