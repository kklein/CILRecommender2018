import os
import random
import math
import numpy as np
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_svd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/svd15_scores.csv')
N_EPOCHS = 15

def write_svd_score(score, k, take_bias):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %s, %f\n' % (k, str(take_bias), score))

def predict_by_svd(data, approximation_rank, take_bias):
    if take_bias:
        imputed_data = utils.predict_bias(data)
    else:
        imputed_data = utils.predict_by_avg(data, True)
    reconstruction = imputed_data
    for epoch_index in range(N_EPOCHS):
        u_embeddings, singular_values, vh_embeddings =\
                np.linalg.svd(reconstruction)
        u_embeddings = u_embeddings[:, 0:approximation_rank]
        singular_values = singular_values[0:approximation_rank]
        vh_embeddings = vh_embeddings[0:approximation_rank, :]
        reconstruction = np.dot(u_embeddings,\
            np.dot(np.diag(singular_values), vh_embeddings))
        if epoch_index < N_EPOCHS - 1:
            reconstruction = utils.impute(data, reconstruction)
    return reconstruction

def main():
    ranks = [i for i in range(3, 30)]
    k = np.random.choice(ranks)
    take_bias = random.choice([True, False])
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    reconstruction = predict_by_svd(masked_data, k, take_bias)
    reconstruction = utils.clip(reconstruction)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME: %f' % rsme)
    write_svd_score(rsme, k, take_bias)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
