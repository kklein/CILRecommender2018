import os
from datetime import datetime

import numpy as np

import utils
import utils_svd
import model_reg_sgd
import model_iterated_svd
import model_sf

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')
ENSEMBLE_INPUT_DIR = 'data/stacking/90/'


def load_predictions_from_files(file_prefix='submission_'):
    path = os.path.join(utils.ROOT_DIR, ENSEMBLE_INPUT_DIR)
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path, i)) and file_prefix in i]
    all_ratings = []
    for file in files:
        print("loading {}".format(file))
        ratings = utils.load_ratings(file)
        ratings = utils.ratings_to_matrix(ratings)
        all_ratings.append(ratings)
    return all_ratings


def compute_mean_predictions(all_ratings):
    if np.sum(np.isnan(all_ratings)) > 0:
        print("Warning: NaNs enountered in compute_mean_predictions")
    for r in all_ratings:
        print("r:", r[2, :15])
    reconstruction = np.nanmean(np.array(all_ratings), axis=0)
    np.nan_to_num(reconstruction, copy=False)
    reconstruction = utils.impute_by_avg(reconstruction, by_row=False)
    return reconstruction


def main():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    mean_predictions = compute_mean_predictions(load_predictions_from_files())
    rmse = utils.compute_rmse(data, mean_predictions)
    print("mean_predictions rmse:", rmse)
    utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    print("Predictions saved in {}".format(SUBMISSION_FILE))


if __name__ == '__main__':
    main()
