import os
from datetime import datetime

import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')


def main():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    mean_predictions = utils.compute_mean_predictions(utils.load_predictions_from_files())
    rmse = utils.compute_rmse(data, mean_predictions)
    print("mean_predictions rmse:", rmse)
    utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    print("Predictions saved in {}".format(SUBMISSION_FILE))


if __name__ == '__main__':
    main()
