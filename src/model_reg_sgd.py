import os
import random
import numpy as np
import utils
import utils_sgd
import model_svd

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sgd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/reg_sgd100_scores.csv')
N_EPOCHS = 100
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.00001

def learn(masked_data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        regularization):
    u_embedding = np.float128(u_embedding)
    z_embedding = np.float128(z_embedding)
    u_bias = np.float128(u_bias)
    z_bias = np.float128(z_bias)
    last_rsme = 5
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(masked_data[np.nonzero(masked_data)])
    for i in range(n_epochs):
        print("Epoch {0}:".format(i))
        random.shuffle(training_indices)
        for k, l in training_indices:
            u_values = u_embedding[k, :]
            z_values = z_embedding[l, :]
            residual = masked_data[k, l] - total_average - u_bias[k] - z_bias[l] -\
                    np.dot(u_values, z_values)
            # residual = masked_data[k, l] - total_average - np.dot(u_values, z_values)
            u_embedding[k, :] *= (1 - regularization * LEARNING_RATE)
            u_embedding[k, :] += LEARNING_RATE * residual * z_values
            z_embedding[l, :] *= (1 - regularization * LEARNING_RATE)
            z_embedding[l, :] += LEARNING_RATE * residual * u_values
            u_bias[k] *= (1 - regularization * LEARNING_RATE)
            u_bias[k] += LEARNING_RATE * residual
            z_bias[l] *= (1 - regularization * LEARNING_RATE)
            z_bias[l] += LEARNING_RATE * residual
        reconstruction = utils_sgd.reconstruct(u_embedding, z_embedding,
                total_average, u_bias, z_bias)
        # reconstruction = np.dot(u_embedding, z_embedding.T) + total_average
        # Training rsme.
        rsme = utils.compute_rsme(masked_data, reconstruction, utils.get_observed_indeces(masked_data))
        print(rsme)
        if abs(last_rsme - rsme) < EPSILON:
            break
        last_rsme = rsme

def predict_by_sgd(masked_data, approximation_rank=None, regularization=REGULARIZATION,
        u_embedding=None, z_embedding=None, n_epochs=N_EPOCHS):
    np.random.seed(42)
    if u_embedding is None and z_embedding is None:
        print("Initialize embeddings.")
        u_embedding, z_embedding = utils_sgd.get_initialized_embeddings(
                approximation_rank, masked_data.shape[0], masked_data.shape[1])
        u_bias, z_bias = utils_sgd.get_initialized_biases(masked_data)
    else:
        print('randomly initialize biases.')
        u_bias = np.random.rand(u_embedding.shape[0])
        z_bias = np.random.rand(z_embedding.shape[0])
    if u_embedding is None or z_embedding is None:
        raise ValueError("embedding is None!")

    learn(masked_data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
            regularization)
    total_average = np.mean(masked_data[np.nonzero(masked_data)])
    reconstruction = utils_sgd.reconstruct(u_embedding, z_embedding, total_average, u_bias, z_bias)
    utils.clip(reconstruction)
    return reconstruction, u_embedding

def main():
    np.seterr(all='raise')
    k = 12
    # k = int(sys.argv[1])
    regularization = 0.2
    # regularization = float(sys.argv[2])
    # ranks = [i for i in range(3, 25)]
    # regularizations = [0.005, 0.002, 0.02, 0.05, 0.2, 0.5]
    # k = np.random.choice(ranks)
    # regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    # svd_initiliazied = random.choice([True, False])
    svd_initiliazied = True
    if svd_initiliazied:
        initialization_string = 'svd'
        imputed_data = np.copy(masked_data)
        utils.impute_by_novel(imputed_data)
        u_embeddings, z_embeddings = model_svd.get_embeddings(imputed_data, k)
        reconstruction, u_embeddings =\
                predict_by_sgd(masked_data, k, regularization, u_embeddings,
                z_embeddings)
    else:
        initialization_string = 'rand'
        reconstruction, u_embeddings =\
                predict_by_sgd(masked_data, k, regularization)
    print(initialization_string)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME before smoothing: %f' % rsme)
    utils_sgd.write_sgd_score(rsme, k, regularization, regularization, '!S',
            initialization_string, SCORE_FILE)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME after smoothing: %f' % rsme)
    utils_sgd.write_sgd_score(rsme, k, regularization, regularization, 'S',
            initialization_string, SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
