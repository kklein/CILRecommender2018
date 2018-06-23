import os
import numpy as np
import model_svd as svd
import model_sf as sf
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/chain.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/chain_scores.csv')

N_META_EPOCHS = 3
N_EPOCHS = 2
REG_EMB = 0.02
REG_BIAS = 0.05
APPROXIMATION_RANK = 4

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
        if np.isnan(u_embeddings).any() or np.isnan(z_embeddings).any():
            raise ValueError('Embeddings contain NaNs.')
        print("Executing sgd by sf.")
        reconstruction = sf.predict_by_sf(masked_data,
                reg_emb=REG_EMB, reg_bias=REG_BIAS, n_epochs=N_EPOCHS,
                u_embedding=u_embeddings, z_embedding=z_embeddings)
        if np.isnan(reconstruction).any():
            raise ValueError('Sf reconstruction created NaNs.')
        utils.ampute_reconstruction(reconstruction, data)
    rsme = utils.compute_rsme(data, reconstruction)
    print(rsme)
    # write_chain_score(SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
