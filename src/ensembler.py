import os
from datetime import datetime

import numpy as np
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor

import utils
import model_reg_sgd

# This file is meant as a template. Feel free to change, replace or copy.
# TODO(b-hahn): Define and use meaningful naming scheme.
SUBMISSION_FILE = os.path.join(
    utils.ROOT_DIR, 'data/ensembles/ensemble' +
    datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')


def run_stacked_svd():
    # k = 10
    # k = int(sys.argv[1])
    # regularization = REGULARIZATION
    # regularization = float(sys.argv[2])
    ranks = [i for i in range(3, 100)]
    regularizations = [0.0005 * i for i in range(400)]
    k = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    # for every fold in training data, train one regressor via SGD and
    # validate on the holdout set. I.e. if we have 4 folds, train 4 different SGD regressors
    # and return each of their reconstructions. For testing, just average those 4 and see what the RMSE is - should be better than a single regressor.

    # write a new version of compute RMSE that doesn't automatically load the validation indices from a file
    reconstruction = []
    lvl2_train_labels = []
    # lvl2_validation_data = []
    # lvl2_validation_labels = []

    num_folds = 4
    kfolds_indices = utils.k_folds(masked_data, num_folds)
    for i, indices in enumerate(kfolds_indices):
        # train using num_folds - 1 folds (i.e. all other folds)
        training_indices = np.hstack([kfolds_indices[j] for j in range(len(kfolds_indices)) if j != i])
        training_folds = np.zeros_like(masked_data)
        training_folds[training_indices] = masked_data[training_indices]

        validation_indices = [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]

        # out_of_fold_prediction = predict_by_sgd(training_folds, k, regularization)
        out_of_fold_prediction = np.zeros_like(masked_data)
        reconstruction.append(out_of_fold_prediction)
        # reconstruction.append(training_folds) # TODO: remove debug code

        lvl2_train_labels.append(masked_data[validation_indices])

        print(lvl2_train_labels)

        rmse = utils.compute_fold_rmse(masked_data, reconstruction[-1], validation_indices)
        rmse_debug = utils.compute_fold_rmse(data, reconstruction[-1],
                                             validation_indices)  # TODO: remove debug code. This should equal 'rmse'
        print("RMSE for fold {}: {}".format(i, rmse))
        utils_sgd.write_sgd_score(rmse, k, regularization, SCORE_FILE)
        utils.reconstruction_to_predictions(reconstruction[-1],
                                            SUBMISSION_FILE.split('.csv')[0] + str(i) + '.csv', validation_indices)

    # TODO: merge above results into one matrix of size num_folds x num_methods
    # to do so, reshape reconstruction gt to n_samples (=n_folds) x n_features (=reconstruction.ravel())
    # Q: need to pass full reconstruction, right?
    reconstruction = np.array([r.ravel() for r in reconstruction])
    print("reconstruction: ", reconstruction.shape)
    print(reconstruction)

    lvl2_train_labels = np.concatenate([r.ravel() for r in lvl2_train_labels])
    print("lvl2_train_labels.shape:", lvl2_train_labels.shape)
    lvl2_train_labels = lvl2_train_labels.ravel()
    print(lvl2_train_labels)

    # TODO: train NN on the matrix. The input are the out-of-fold predictions by each model,
    # the labels are the ground truth ratings from the holdout fold.
    regressor = MLPRegressor(hidden_layer_sizes=(100, 100))
    regressor.fit(X=reconstruction, y=lvl2_train_labels)

    # Once the model has been fit, run each model (or only reg_sgd) on the validation data and use what
    # it outputs as the input for the NN. Get output and calculate RMSE.
    lvl2_predictions = regressor.predict(X=data)
    print(lvl2_predictions)

#     reconstruction = predict_by_sgd(masked_data, k, regularization)
#     rsme = utils.compute_rsme(data, reconstruction)


def bagging(n):
    """
    Implements simple bagging for CF: generate n matrices of same size as data by 
    sampling rows with replacement. Run prediction on each, combine predictions. This
    should improve results since it reduces overall variance.
    """
    ranks = [i for i in range(3, 100)]
    regularizations = [0.0005 * i for i in range(400)]
    k = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)


    k = 3  # TODO: remove for real runs
    predictions = []
    sampled_data = np.zeros_like(masked_data)
    sampled_users = np.zeros(masked_data.shape[0])
    # TODO: make sure we're sampling users and not items
    print(sampled_users.shape)
    for i in range(n):
        for r in range(masked_data.shape[0]):
            random_row = np.random.choice(masked_data.shape[0])
            # keep track of which user (i.e. row) is added. Later, average ratings of duplicates of each user
            sampled_users[r] = random_row
            sampled_data[r, :] = masked_data[random_row, :]
        sampled_prediction = model_reg_sgd.predict_by_sgd(sampled_data, k, regularization)
        # sort predictions by user and group duplicates
        # then calculate mean predictions for duplicates.
        prediction = np.zeros_like(masked_data)
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
    bagging(3)


if __name__ == '__main__':
    main()
