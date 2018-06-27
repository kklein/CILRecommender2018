import os
import sys
from datetime import datetime
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
    utils.impute_by_variance(reconstruction)
    print('Initial imputation completed.')
    for i in range(n_meta_epochs):
        print("Computing embeddings.")
        u_embeddings, z_embeddings = svd.get_embeddings(
            reconstruction, approximation_rank)
        print("Executing sgd by sf.")
        predictions, u_embeddings, z_embeddings = sf.predict_by_sf(
            masked_data, reg_emb=REG_EMB, reg_bias=REG_BIAS, n_epochs=N_EPOCHS,
            u_embeddings=u_embeddings, z_embeddings=z_embeddings)
        reconstruction = np.dot(u_embeddings, z_embeddings.T)
        utils.ampute_reconstruction(reconstruction, masked_data)
        rmse = utils.compute_rmse(data, reconstruction)
        print("meta iteration %d val. rmse: %f" % (i, rmse))
    reconstruction = utils.knn_smoothing(predictions, u_embeddings)
    rmse = utils.compute_rmse(data, reconstruction)
    print("rmse after smoothing: %f" % rmse)
    write_chain_score(approximation_rank, n_meta_epochs, rmse)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)
    utils.reconstruction_to_predictions(
        reconstruction,
        utils.ROOT_DIR + 'data/meta_training_chain_svd_stacking' +
        datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_first.csv"))
    utils.reconstruction_to_predictions(
        reconstruction,
        utils.ROOT_DIR + 'data/meta_validation_chain_svd_stacking' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') +
        '.csv',
        indices_to_predict=utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_second.csv"))

if __name__ == '__main__':
    main()
