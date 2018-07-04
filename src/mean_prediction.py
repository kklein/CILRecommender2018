import os
from datetime import datetime

import numpy as np

import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')
ENSEMBLE_INPUT_DIR = 'data/stacking/90/'


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
    mean_predictions = compute_mean_predictions(utils.load_predictions_from_files())
    rmse = utils.compute_rmse(data, mean_predictions)
    print("mean_predictions rmse:", rmse)
    utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    print("Predictions saved in {}".format(SUBMISSION_FILE))


if __name__ == '__main__':
    main()
