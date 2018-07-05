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
    for _ in range(N_EPOCHS):
        u_embeddings, z_embeddings =\
                svd.get_embeddings(reconstruction, approximation_rank)
        reconstruction = np.matmul(u_embeddings, z_embeddings.T)
        utils.ampute_reconstruction(reconstruction, data)
    reconstruction = utils.clip(reconstruction)
    return reconstruction, u_embeddings, z_embeddings

def main():
    ranks = [i for i in range(3, 25)]
    rank = np.random.choice(ranks)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data, False)
    imputed_data = np.copy(masked_data)
    utils.impute_by_avg(imputed_data, True)
    reconstruction, u_embeddings, _ =\
            predict_by_svd(masked_data, imputed_data, rank)
    rmse = utils.compute_rmse(data, reconstruction)
    print('rmse before smoothing: %f' % rmse)
    svd.write_svd_score(rmse, rank, False, SCORE_FILE)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rmse = utils.compute_rmse(data, reconstruction)
    print('rmse after smoothing: %f' % rmse)
    svd.write_svd_score(rmse, rank, True, SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)
    if utils.SAVE_META_PREDICTIONS:
        utils.save_ensembling_predictions(reconstruction, 'iterated_svd')

if __name__ == '__main__':
    main()
