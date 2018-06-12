import os
import random
import numpy as np
import utils
import utils_sgd
import model_svd

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sf_sgd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/sf100_sgd_scores.csv')
N_EPOCHS = 5
LEARNING_RATE = 0.001
REG_EMB = 0.02
REG_BIAS = 0.05
EPSILON = 0.0001

def learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        reg_emb, reg_bias):
    residual_data = data

    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(data[np.nonzero(data)])
    approximation_rank = u_embedding.shape[1]
    for feature_index in range(approximation_rank):
        print("Feature %d." % feature_index)
        last_rsme = 5
        for i in range(n_epochs):
            print("Epoch {0}:".format(i))
            random.shuffle(training_indices)

            for k, l in training_indices:
                u_value = u_embedding[k, feature_index]
                z_value = z_embedding[l, feature_index]
                residual = residual_data[k, l] - total_average - u_bias[k] -\
                        z_bias[l] - u_value * z_value
                u_update =\
                        LEARNING_RATE * (residual * z_value - reg_emb * u_value)
                z_update =\
                        LEARNING_RATE * (residual * u_value - reg_emb * z_value)
                u_bias_update =\
                        LEARNING_RATE * (residual - reg_bias * u_bias[k])
                z_bias_update =\
                        LEARNING_RATE * (residual - reg_bias * z_bias[l])
                u_embedding[k, feature_index] += u_update
                z_embedding[l, feature_index] += z_update
                u_bias[k] += u_bias_update
                z_bias[l] += z_bias_update

            reconstruction = utils_sgd.reconstruct(
                    u_embedding[:, :feature_index + 1],
                    z_embedding[:, :feature_index + 1], total_average, u_bias,
                    z_bias)
            residual_data = data - reconstruction
            rsme = utils.compute_rsme(data, reconstruction)
            print(rsme)
            if abs(last_rsme - rsme) < EPSILON:
                break
            last_rsme = rsme

            if np.isnan(u_embedding).any() or np.isnan(z_embedding).any():
                raise ValueError('Found NaN in embedding after feature %d.' % feature_index)

# sf stands for Simon Funk.
def predict_by_sf(data, approximation_rank=None, reg_emb=REG_EMB,
        reg_bias=REG_BIAS, u_embedding=None, z_embedding=None,
        n_epochs=N_EPOCHS):
    np.random.seed(42)
    if u_embedding is None and z_embedding is None:
        print("Initialize embeddings.")
        u_embedding, z_embedding = utils_sgd.get_initialized_embeddings(
                approximation_rank, data.shape[0], data.shape[1])
    if u_embedding is None or z_embedding is None:
        raise ValueError("embedding is None!")
    u_bias, z_bias = utils_sgd.get_initialized_biases(data)
    learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        reg_emb, reg_bias)
    total_average = np.mean(data[np.nonzero(data)])
    reconstruction = utils_sgd.reconstruct(u_embedding, z_embedding, total_average, u_bias, z_bias)
    utils.clip(reconstruction)
    return reconstruction, u_embedding

def main():
    # k = int(sys.argv[1])
    # regularization = float(sys.argv[2])
    ranks = [i for i in range(3, 40)]
    regularizations = [0.005, 0.002, 0.02, 0.05, 0.2, 0.5]
    reg_emb = np.random.choice(regularizations)
    reg_bias = np.random.choice(regularizations)
    k = np.random.choice(ranks)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    # svd_initiliazied = random.choice([True, False])
    svd_initiliazied = False
    if svd_initiliazied:
        initialization_string = 'svd'
        imputed_data = utils.impute_by_novel(masked_data)
        u_embeddings, z_embeddings = model_svd.get_embeddings(imputed_data, k)
        reconstruction, u_embeddings =\
                predict_by_sf(masked_data, k, reg_emb, reg_bias, u_embeddings,
                z_embeddings)
    else:
        initialization_string = 'rand'
        reconstruction, u_embeddings =\
                predict_by_sf(masked_data, k, reg_emb, reg_bias)

    rsme = utils.compute_rsme(data, reconstruction)
    print('Validation RSME before smoothing: %f' % rsme)
    utils_sgd.write_sgd_score(rsme, k, reg_emb, reg_bias, '!S',
            initialization_string, SCORE_FILE)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    print('Validation RSME after smoothing: %f' % rsme)
    utils_sgd.write_sgd_score(rsme, k, reg_emb, reg_bias, 'S',
            initialization_string, SCORE_FILE)

    # utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
