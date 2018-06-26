import os
import sys
import numpy as np
import utils_svd as svd
import model_sf as sf
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/chain.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/chain_scores.csv')

N_META_EPOCHS = 6
N_EPOCHS = 100
REG_EMB = 0.02
REG_BIAS = 0.05
APPROXIMATION_RANK = 20

def write_chain_score(approximation_rank, n_meta_epochs, score):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %d, %f\n' % (approximation_rank, n_meta_epochs, score))

def main():
    if len(sys.argv) == 1:
        approximation_rank = APPROXIMATION_RANK
        n_meta_epochs = N_META_EPOCHS
    else:
        approximation_rank = int(sys.argv[1])
        n_meta_epochs = int(sys.argv[2])
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    reconstruction = np.copy(masked_data)
    utils.impute_by_novel(reconstruction)
    print('Initial imputation completed.')
    for i in range(n_meta_epochs):
        print("Computing embeddings.")
        u_embeddings, z_embeddings =\
                svd.get_embeddings(reconstruction, approximation_rank)
        print("Executing sgd by sf.")
        predictions, u_embeddings, z_embeddings = sf.predict_by_sf(masked_data,
                reg_emb=REG_EMB, reg_bias=REG_BIAS, n_epochs=N_EPOCHS,
                u_embedding=u_embeddings, z_embedding=z_embeddings)
        reconstruction = np.dot(u_embeddings, z_embeddings.T)
        utils.ampute_reconstruction(reconstruction, masked_data)
        rsme = utils.compute_rsme(data, reconstruction)
        print("meta iteration %d val. rsme: %f" % (i, rsme))
    reconstruction = utils.knn_smoothing(predictions, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    print("RSME after smoothing: %f" % rsme)
    write_chain_score(approximation_rank, n_meta_epochs, rsme)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
