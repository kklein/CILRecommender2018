from random import shuffle
import os
import numpy as np
import utils
import utils_sgd


SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sf_sgd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/sf15_sgd_scores.csv')
N_EPOCHS = 15
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.0001

def learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        regularization):

    if isinstance(regularization, int):
        regularization_1 = regularization
        regularization_2 = regularization
    else:
        regularization_1, regularization_2 = regularization

    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(data[np.nonzero(data)])
    approximation_rank = u_embedding.shape[1]
    for feature_index in range(approximation_rank):
        print("Feature %d." % feature_index)
        for i in range(n_epochs):
            print("Epoch {0}:".format(i))
            shuffle(training_indices)
            for k, l in training_indices:
                residual = data[k, l] - total_average - u_bias[k] - z_bias[l]\
                        - np.dot(u_embedding[k, :feature_index + 1],\
                        z_embedding[l, :feature_index + 1])
                u_update = LEARNING_RATE * residual * z_embedding[l, feature_index] - \
                        utils.safe_norm(LEARNING_RATE * regularization_1 * \
                        u_embedding[k, feature_index])
                z_update = LEARNING_RATE * residual * u_embedding[k, feature_index] - \
                        utils.safe_norm(LEARNING_RATE * regularization_1 * \
                        z_embedding[l, feature_index])
                u_bias_update = LEARNING_RATE * (residual - regularization_2 *
                        np.absolute(u_bias[k]))
                z_bias_update = LEARNING_RATE * (residual - regularization_2 *
                        np.absolute(z_bias[l]))
                u_embedding[k, feature_index] += u_update
                z_embedding[l, feature_index] += z_update
                u_bias[k] += u_bias_update
                z_bias[l] += z_bias_update

# sf stands for Simon Funk.
def predict_by_sf(data, approximation_rank=None, regularization=REGULARIZATION,
        n_epochs=N_EPOCHS, u_embedding=None, z_embedding=None):
    np.random.seed(42)
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
    reconstruction = predict_by_sf(masked_data, k, regularization)
    rsme = utils.compute_rsme(data, reconstruction)
    utils_sgd.write_sgd_score(rsme, k, regularization, SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
