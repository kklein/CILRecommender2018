import os
import random
import numpy as np
import utils
import utils_sgd
import utils_svd as svd

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/submission_reg_sgd.csv')

SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/reg_sgd100_scores.csv')
N_EPOCHS = 100
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.00001

def learn(masked_data, u_embeddings, z_embeddings, u_bias, z_bias, n_epochs,
          regularization):
    last_rmse = 5
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(masked_data[np.nonzero(masked_data)])
    for i in range(n_epochs):
        print("Epoch {0}:".format(i))
        random.shuffle(training_indices)
        for k, l in training_indices:
            lagrangian = u_bias[k] + z_bias[l] - total_average
            u_values = u_embeddings[k, :]
            z_values = z_embeddings[l, :]
            residual = masked_data[k, l] - u_bias[k] - z_bias[l] - np.dot(u_values, z_values)
            # residual = masked_data[k, l] - total_average - np.dot(u_values, z_values)
            u_embeddings[k, :] *= (1 - regularization * LEARNING_RATE)
            u_embeddings[k, :] += LEARNING_RATE * residual * z_values
            z_embeddings[l, :] *= (1 - regularization * LEARNING_RATE)
            z_embeddings[l, :] += LEARNING_RATE * residual * u_values
            u_bias[k] -= regularization * LEARNING_RATE * lagrangian
            u_bias[k] += LEARNING_RATE * residual
            z_bias[l] -= regularization * LEARNING_RATE * lagrangian
            z_bias[l] += LEARNING_RATE * residual
        reconstruction = utils_sgd.reconstruct(
            u_embeddings, z_embeddings, u_bias, z_bias)
        # reconstruction = np.dot(u_embeddings, z_embeddings.T) + total_average
        # Training rmse.
        rmse = utils.compute_rmse(
            masked_data, reconstruction,
            utils.get_observed_indeces(masked_data))
        print(rmse)
        if abs(last_rmse - rmse) < EPSILON:
            break
        last_rmse = rmse
    return reconstruction

def predict_by_sgd(masked_data, approximation_rank=None,
                   regularization=REGULARIZATION, u_embeddings=None,
                   z_embeddings=None, n_epochs=N_EPOCHS):
    np.random.seed(42)

    if u_embeddings is None and z_embeddings is None:
        print("Initialize embeddings.")
        u_embeddings, z_embeddings = utils_sgd.get_initialized_embeddings(
            approximation_rank, masked_data.shape[0], masked_data.shape[1])
    u_bias = np.zeros(u_embeddings.shape[0])
    z_bias = np.zeros(z_embeddings.shape[0])
    reconstruction = learn(
        masked_data, u_embeddings, z_embeddings, u_bias, z_bias, n_epochs,
        regularization)
    utils.clip(reconstruction)
    return reconstruction, u_embeddings, z_embeddings

def main():
    np.seterr(all='raise')
    ranks = [i for i in range(3, 25)]
    regularizations = [0.01, 0.02, 0.03, 0.05, 0.1]
    rank = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)

    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data, False)
    svd_initiliazied = random.choice([True, False])

    if svd_initiliazied:
        initialization_string = 'svd'
        imputed_data = np.copy(masked_data)
        utils.impute_by_variance(imputed_data)
        u_embeddings, z_embeddings = svd.get_embeddings(imputed_data, rank)
        reconstruction, u_embeddings, _ = predict_by_sgd(
            masked_data, rank, regularization, u_embeddings, z_embeddings)
    else:
        initialization_string = 'rand'
        reconstruction, u_embeddings, _ = predict_by_sgd(
            masked_data, rank, regularization)

    rmse = utils.compute_rmse(data, reconstruction)
    print('rmse before smoothing: %f' % rmse)
    utils_sgd.write_sgd_score(
        rmse, rank, regularization, regularization, '!S', initialization_string,
        SCORE_FILE)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rmse = utils.compute_rmse(data, reconstruction)
    print('rmse after smoothing: %f' % rmse)
    utils_sgd.write_sgd_score(
        rmse, rank, regularization, regularization, 'S', initialization_string, SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)
    if utils.SAVE_META_PREDICTIONS:
        utils.save_ensembling_predictions(reconstruction, 'reg_svd')

if __name__ == '__main__':
    main()
