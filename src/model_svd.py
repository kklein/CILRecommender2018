import os
import random
import numpy as np
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_svd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/svd15_scores.csv')
N_EPOCHS = 15

def write_svd_score(score, k, take_bias):
    with open(SCORE_FILE, 'a+') as file:
        file.write('%d, %s, %f\n' % (k, str(take_bias), score))

def get_embeddings(imputed_data, approximation_rank):
    u_embeddings, singular_values, z_embeddings =\
            np.linalg.svd(imputed_data)
    u_embeddings = np.dot(u_embeddings, np.sqrt(singular_values))
    z_embeddings = np.dot(z_embeddings, np.sqrt(singular_values))
    u_embeddings = u_embeddings[:, 0:approximation_rank]
    z_embeddings = z_embeddings[0:approximation_rank, :]
    return u_embeddings, z_embeddings

def predict_by_svd(data, imputed_data, approximation_rank,):
    reconstruction = imputed_data
    for epoch_index in range(N_EPOCHS):
        u_embeddings, z_embeddings =\
                get_embeddings(data, imputed_data, approximation_rank)
        reconstruction = np.dot(u_embeddings, z_embeddings)
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
    if take_bias:
        imputed_data = utils.predict_bias(data)
    else:
        imputed_data = utils.predict_by_avg(data, True)
    reconstruction = predict_by_svd(masked_data, imputed_data, k, take_bias)
    reconstruction = utils.clip(reconstruction)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME: %f' % rsme)
    write_svd_score(rsme, k, take_bias)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
