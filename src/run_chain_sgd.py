import os
import numpy as np
import model_svd as svd
import model_reg_sgd as sgd
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/chain.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/chain_scores.csv')

N_META_EPOCHS = 6
N_EPOCHS = 100
REG_EMB = 0.02
REG_BIAS = 0.05
APPROXIMATION_RANK = 20

def main():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    reconstruction = np.copy(masked_data)
    utils.impute_by_novel(reconstruction)
    print('Initial imputation completed.')
    for _ in range(N_META_EPOCHS):
        print("Computing embeddings.")
        u_embeddings, z_embeddings =\
                svd.get_embeddings(reconstruction, APPROXIMATION_RANK)
        print("Executing reg sgd.")
        reconstruction, u_embeddings = sgd.predict_by_sgd(masked_data,
                regularization=REG_EMB, n_epochs=N_EPOCHS,
                u_embedding=u_embeddings, z_embedding=z_embeddings)
        utils.ampute_reconstruction(reconstruction, masked_data)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    print(rsme)
    # write_chain_score(SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
