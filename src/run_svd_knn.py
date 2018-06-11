import os
import numpy as np
import model_svd
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_svd_knn.csv')
APPROXIMATION_RANK = 15

def main():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    imputed_data = utils.predict_by_avg(data, True)
    reconstruction, u_embeddings, _ =\
            model_svd.predict_by_svd(
            masked_data, imputed_data, APPROXIMATION_RANK)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME before knn: %f' % rsme)
    reconstruction = utils.knn_smoothing(reconstruction, u_embeddings)
    rsme = utils.compute_rsme(data, reconstruction)
    print('RSME after knn: %f' % rsme)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
