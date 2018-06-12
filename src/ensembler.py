import os
from datetime import datetime

import numpy as np
from sklearn import ensemble

import utils
import model_reg_sgd

# This file is meant as a template. Feel free to change, replace or copy.
# TODO(b-hahn): Define and use meaningful naming scheme.
SUBMISSION_FILE = os.path.join(
    utils.ROOT_DIR, 'data/ensembles/ensemble' +
    datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')


def bagging(n):
    """
    Implements simple bagging for CF: generate n matrices of same size as data by 
    sampling rows with replacement. Run prediction on each, combine predictions. This
    should improve results since it reduces overall variance.
    """
    data, masked_data, k, regularization = model_reg_sgd.init_reg_sgd()

    predictions = []
    sampled_data = np.zeros_like(data)
    sampled_users = np.zeros(data.shape[0])
    # TODO: make sure we're sampling users and not items
    print(sampled_users.shape)
    for i in range(n):
        for r in range(data.shape[0]):
            random_row = np.random.choice(data.shape[0])
            # keep track of which user (i.e. row) is added. Later, average ratings of duplicates of each user
            sampled_users[r] = random_row
            sampled_data[r, :] = data[random_row, :]
        sampled_prediction = model_reg_sgd.predict_by_sgd(sampled_data, k, model_reg_sgd.REGULARIZATION)
        # sort predictions by user and group duplicates
        # then calculate mean predictions for duplicates.
        prediction = np.zeros_like(data)
        nan_count = 0
        for r in range(prediction.shape[0]):
            # mean of rows in sampled_prediction where entries in sampled_users equal r
            prediction[r, :] = np.mean(sampled_prediction[np.argwhere(sampled_users == r), :], axis=0)
            # print("NaNs in row {}: {}".format(r, np.sum(np.isnan(prediction[r, :]))))
            if np.sum(np.isnan(prediction[r, :])) > 0:
                nan_count += 1
        print("nan_count:", nan_count)
        np.nan_to_num(prediction, copy=False)
        print(prediction.shape, np.sum(np.isnan(prediction)))
        predictions.append(prediction)

    print("Finished {} runs of bagging...calculating mean of predictions".format(n))
    compute_mean_predictions(predictions)


def load_predictions_from_files():
    path = os.path.join(utils.ROOT_DIR, 'data/ensembles/4folds')
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path,i)) and 'submission_' in i]
    all_ratings = []
    for n, file in enumerate(files):
        print("loading {}".format(file))
        ratings = utils.load_ratings(file)
        ratings = utils.ratings_to_matrix(ratings)
        all_ratings.append(ratings)
    return all_ratings


def compute_mean_predictions(all_ratings):
    reconstruction = np.mean(np.array(all_ratings), axis=0)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)
    print("Predictions saved in {}".format(SUBMISSION_FILE))


def main():
    # compute_mean_predictions(load_predictions_from_files())
    bagging(2)


if __name__ == '__main__':
    main()
