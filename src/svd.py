import sys
import os
import numpy as np
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_svd.csv')

def predict_by_svd(data, approximation_rank):
    imputed_data = utils.predict_by_avg(data, True)
    # imputed_data = utils.predict_bias(data)
    u_embeddings, singular_values, vh_embeddings =\
            np.linalg.svd(imputed_data)
    u_embeddings = u_embeddings[:, 0:approximation_rank]
    singular_values = singular_values[0:approximation_rank]
    vh_embeddings = vh_embeddings[0:approximation_rank, :]
    return np.dot(u_embeddings,\
            np.dot(np.diag(singular_values), vh_embeddings))

def main():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    reconstruction = predict_by_svd(data, 10)
    reconstruction = utils.clip(reconstruction)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME: %f' % rsme)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
