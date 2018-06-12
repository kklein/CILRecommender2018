import os
import random
import numpy as np
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_svd30_7.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/svd30_scores.csv')
N_EPOCHS = 30

def write_svd_score(score, k, take_bias):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %s, %f\n' % (k, str(take_bias), score))

def get_embeddings(imputed_data, approximation_rank):
    u_embeddings, singular_values, z_embeddings =\
            np.linalg.svd(imputed_data)
    u_embeddings = u_embeddings[:, 0:approximation_rank]
    z_embeddings = z_embeddings[0:approximation_rank, :]
    singular_values = singular_values[:approximation_rank]
    s = np.zeros((approximation_rank, approximation_rank))
    np.fill_diagonal(s, np.sqrt(singular_values))
    u_embeddings = np.matmul(u_embeddings, s)
    z_embeddings = np.matmul(s, z_embeddings)
    z_embeddings = z_embeddings.T
    return u_embeddings, z_embeddings

def predict_by_svd(data, imputed_data, approximation_rank,):
    reconstruction = imputed_data
    for epoch_index in range(N_EPOCHS):
        u_embeddings, z_embeddings =\
                get_embeddings(imputed_data, approximation_rank)
        reconstruction = np.matmul(u_embeddings, z_embeddings.T)
        if epoch_index < N_EPOCHS - 1:
            reconstruction = utils.impute(data, reconstruction)
    reconstruction = utils.clip(reconstruction)
    return reconstruction, u_embeddings, z_embeddings

def main():
    # ranks = [i for i in range(3, 25)]
    # k = np.random.choice(ranks)
    k = 7
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    imputed_data = utils.impute_by_avg(data, True)
    reconstruction, u_embeddings, _ =\
            predict_by_svd(masked_data, imputed_data, k)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME before smoothing: %f' % rsme)
    # write_svd_score(rsme, k, False)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)
    # print('RSME before after smoothing: %f' % rsme)
    # write_svd_score(rsme, k, True)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
