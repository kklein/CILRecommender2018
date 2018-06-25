import os
import numpy as np
import utils
import utils_svd as svd

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_svd30_7.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/svd30_scores.csv')
N_EPOCHS = 20

def predict_by_svd(data, imputed_data, approximation_rank):
    reconstruction = imputed_data
    for epoch_index in range(N_EPOCHS):
        u_embeddings, z_embeddings =\
                svd.get_embeddings(reconstruction, approximation_rank)
        reconstruction = np.matmul(u_embeddings, z_embeddings.T)
        utils.ampute_reconstruction(reconstruction, data)
    reconstruction = utils.clip(reconstruction)
    return reconstruction, u_embeddings, z_embeddings

def main():
    ranks = [i for i in range(3, 25)]
    k = np.random.choice(ranks)
    # k = 4
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    imputed_data = np.copy(masked_data)
    utils.impute_by_avg(imputed_data, True)
    reconstruction, u_embeddings, _ =\
            predict_by_svd(masked_data, imputed_data, k)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME before smoothing: %f' % rsme)
    svd.write_svd_score(rsme, k, False, SCORE_FILE)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)
    print('RSME after smoothing: %f' % rsme)
    svd.write_svd_score(rsme, k, True, SCORE_FILE)
    # utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
