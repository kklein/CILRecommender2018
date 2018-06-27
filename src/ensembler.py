import os
from datetime import datetime

import numpy as np
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

import utils
import model_reg_sgd
import model_svd

# This file is meant as a template. Feel free to change, replace or copy.
# TODO(b-hahn): Define and use meaningful naming scheme.
SUBMISSION_FILE = os.path.join(
    utils.ROOT_DIR, 'data/ensemble' +
    datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')
ENSEMBLE_INPUT_DIR = 'data/stacking'
STACKING_METHOD = 'nn'


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

def generate_debug_predictions():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    # TODO: make sure the mask is on 20% not 10%
    masked_data = utils.mask_validation(data)


def stacking(meta_training, meta_validation):
    ground_truth_ratings = utils.load_ratings()
    ground_truth_ratings = utils.ratings_to_matrix(ground_truth_ratings)
    # TODO: rename function to something more appropriate?

    train_indices = utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_first.csv")
    train_ratings_predictions = np.squeeze([[rating[i, j] for rating in meta_training] for i, j in train_indices])
    train_ratings_target = [ground_truth_ratings[i, j] for i, j in train_indices]
    validation_indices = utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_second.csv")
    validation_ratings_predictions = np.squeeze(
        [[rating[i, j] for rating in meta_validation] for i, j in validation_indices])
    # validation_ratings_target = [ground_truth_ratings[i, j] for i, j in validation_indices]

    if STACKING_METHOD == 'lr':
        # using linear regression
        weights, res, _, _ = np.linalg.lstsq(train_ratings_predictions, train_ratings_target)
        print("Weights: {}\tres: {}".format(weights, res))
        lvl2_predictions = np.dot(weights, validation_ratings_predictions.T)

    elif STACKING_METHOD == 'pr':
        # using polynomial regression
        weights, res, _, _ = np.polyfit(train_ratings_predictions, train_ratings_target, deg=2, full=True)
        print("Weights: {}\tres: {}".format(weights, res))
        lvl2_predictions = np.dot(weights, validation_ratings_predictions.T)

    elif STACKING_METHOD == 'nn':
        # using a neural net
        regressor = MLPRegressor(hidden_layer_sizes=(50, 100))
        regressor.fit(X=train_ratings_predictions, y=train_ratings_target)
        lvl2_predictions = regressor.predict(X=validation_ratings_predictions)

    elif STACKING_METHOD == "xgb":
        # using xgboost
        regressor = xgb.XGBRegressor(max_depth=4, learning_rate=0.8, n_estimators=100, eta=0.99)
        regressor.fit(train_ratings_predictions, train_ratings_target)
        lvl2_predictions = regressor.predict(validation_ratings_predictions)

    elif STACKING_METHOD == "kr":
        # kernel ridge regression
        regressor = KernelRidge()
        regressor.fit(train_ratings_predictions, train_ratings_target)
        lvl2_predictions = regressor.predict(validation_ratings_predictions)

    lvl2_predictions = utils.ratings_to_matrix([(validation_indices[i][0], validation_indices[i][1],
                                                 lvl2_predictions[i]) for i in range(len(validation_indices))])
    print(lvl2_predictions[:10])
    lvl2_predictions = utils.clip(lvl2_predictions)

    rmse = utils.compute_rsme(ground_truth_ratings, lvl2_predictions, validation_indices)
    print("lvl2_predictions rmse:", rmse)
    utils.reconstruction_to_predictions(lvl2_predictions, SUBMISSION_FILE, validation_indices)
    print("Predictions saved in {}".format(SUBMISSION_FILE))
    utils.reconstruction_to_predictions(ground_truth_ratings, utils.ROOT_DIR + "gt_data.csv", validation_indices)


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

    predictor = 'svd'
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
        if predictor == 'reg_sgd':
            sampled_prediction, _ = model_reg_sgd.predict_by_sgd(sampled_data, k, regularization)
        elif predictor == 'svd':
            imputed_data = np.copy(sampled_data)
            utils.impute_by_avg(imputed_data, True)
            sampled_prediction, _, _ = model_svd.predict_by_svd(sampled_data, imputed_data, k)
        # sort predictions by user and group duplicates
        # then calculate mean predictions for duplicates.
        prediction = np.zeros_like(masked_data) * np.nan
        nan_count = 0
        for r in range(prediction.shape[0]):
            # mean of rows in sampled_prediction where entries in sampled_users equal r
            # TODO: why are there so many NaNs here?
            duplicate_user_predictions = sampled_prediction[np.argwhere(sampled_users == r), :]
            if duplicate_user_predictions.shape[0] > 0:
                # print("For row {} calculating mean of {} duplicates".format(r, duplicate_user_predictions.shape))
                # print(duplicate_user_predictions[:, 0, :10])
                prediction[r, :] = np.mean(duplicate_user_predictions, axis=0)
            # print("NaNs in row {}: {}".format(r, np.sum(np.isnan(prediction[r, :]))))
            if np.sum(np.isnan(prediction[r, :])) > 0:
                nan_count += 1
        print("nan_count:", nan_count)
        # np.nan_to_num(prediction, copy=False)
        print(prediction.shape, np.sum(np.isnan(prediction)))
        predictions.append(prediction)

    print("Finished {} runs of bagging...calculating mean of predictions".format(n))
    #    print(np.unique(np.concatenate(predictions, axis=0), axis=0).shape[0])
    print(np.sum(np.count_nonzero(np.sum(predictions, axis=0), axis=1) > 0))
    mean_predictions = compute_mean_predictions(predictions)
    rmse = utils.compute_rsme(data, mean_predictions)
    print("mean_predictions rmse:", rmse)
    utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    print("Predictions saved in {}".format(SUBMISSION_FILE))
    utils.reconstruction_to_predictions(
        mean_predictions,
        utils.ROOT_DIR + 'data/meta_training_bagging_svd_stacking' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=utils.get_validation_indices(utils.ROOT_DIR + "data/train_valid_80_10_10/validationIndices_first.csv"))
    utils.reconstruction_to_predictions(
        mean_predictions,
        utils.ROOT_DIR + 'data/meta_validation_bagging_svd_stacking' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') +
        '.csv',
        indices_to_predict=utils.get_validation_indices(
            utils.ROOT_DIR + "data/train_valid_80_10_10/validationIndices_second.csv"))


def load_predictions_from_files(file_prefix='submission_'):
    path = os.path.join(utils.ROOT_DIR, ENSEMBLE_INPUT_DIR)
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path,i)) and file_prefix in i]
    all_ratings = []
    for n, file in enumerate(files):
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
    # TODO(ben): set remaining NaNs to a sensible value, e.g. row mean/ column mean
    np.nan_to_num(reconstruction, copy=False)
    reconstruction = utils.impute_by_avg(reconstruction, by_row=False)
    return reconstruction


def main():
    # all_ratings = utils.load_ratings()
    # data = utils.ratings_to_matrix(all_ratings)
    # mean_predictions = compute_mean_predictions(load_predictions_from_files())
    # rmse = utils.compute_rsme(data, mean_predictions)
    # print("mean_predictions rmse:", rmse)
    # utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    # print("Predictions saved in {}".format(SUBMISSION_FILE))

    # bagging(3)

    stacking(load_predictions_from_files('meta_training'),
             load_predictions_from_files('meta_validation'))


if __name__ == '__main__':
    main()
