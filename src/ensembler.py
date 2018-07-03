import os
from datetime import datetime

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import xgboost as xgb

import utils
import utils_svd
import model_reg_sgd
import model_iterated_svd
import model_sf

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')
ENSEMBLE_INPUT_DIR = 'data/stacking/good_data/'
STACKING_METHOD = 'lr'
BAGGING_METHOD = 'svd'


def stacking(meta_training, meta_validation):
    """
    Stacks/blends the predictions from all models and fits a model to the data 
    in argument meta_training. Validation is done on data in meta_validation. Note 
    that it is assumed that data is split into three parts, where meta_training 
    equals the validation data for the base predictors and meta_validation equals 
    hold-out data not seen by the base predictors.

    :param meta_training:
    :param meta_validation:
    :return:
    """
    ground_truth_ratings = utils.load_ratings()
    ground_truth_ratings = utils.ratings_to_matrix(ground_truth_ratings)

    train_indices = utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_first.csv")
    train_ratings_predictions = np.squeeze([[rating[i, j] for rating in meta_training] for i, j in train_indices])
    train_ratings_target = [ground_truth_ratings[i, j] for i, j in train_indices]
    validation_indices = utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_second.csv")
    validation_ratings_predictions = np.squeeze(
        [[rating[i, j] for rating in meta_validation] for i, j in validation_indices])
    # validation_ratings_target = [ground_truth_ratings[i, j] for i, j in validation_indices]
    test_indices = utils.get_indices_to_predict()
    test_ratings_predictions = load_predictions_from_files("sub")
    test_ratings_predictions = np.squeeze(
        [[rating[i, j] for rating in test_ratings_predictions] for i, j in test_indices])

    if STACKING_METHOD == 'lr':
        # using linear regression
        weights, res, _, _ = np.linalg.lstsq(train_ratings_predictions, train_ratings_target)
        lvl2_validation = np.dot(weights, validation_ratings_predictions.T)
        lvl2_test = np.dot(weights, test_ratings_predictions.T)
    # TODO: could try lr with predictions split into bins according to movie/ user support
    # (i.e. separetly weight users with many ratings vs those with few ratings)

    elif STACKING_METHOD == 'nn':
        # using a neural net
        regressor = MLPRegressor(hidden_layer_sizes=(200, ))
        regressor.fit(X=train_ratings_predictions, y=train_ratings_target)
        lvl2_validation = regressor.predict(X=validation_ratings_predictions)
        lvl2_test = regressor.predict(X=test_ratings_predictions)

    elif STACKING_METHOD == "xgb":
        # using xgboost
        regressor = xgb.XGBRegressor(max_depth=8, learning_rate=0.02, n_estimators=30, eta=0.99)
        regressor.fit(train_ratings_predictions, train_ratings_target)
        lvl2_validation = regressor.predict(validation_ratings_predictions)
        lvl2_test = regressor.predict(test_ratings_predictions)

    elif STACKING_METHOD == "krr":
        # kernel ridge regression
        regressor = KernelRidge(solver='lsqr')
        regressor.fit(train_ratings_predictions, train_ratings_target)
        lvl2_validation = regressor.predict(validation_ratings_predictions)
        lvl2_test = regressor.predict(test_ratings_predictions)
        print("Implement lvl2_test!")
        return

    elif STACKING_METHOD == "rr":
        # ridge regression
        regressor = Ridge(solver='auto', normalize='False')
        regressor.fit(train_ratings_predictions, train_ratings_target)
        lvl2_validation = regressor.predict(validation_ratings_predictions)
        lvl2_test = regressor.predict(test_ratings_predictions)

    lvl2_validation = utils.ratings_to_matrix([(validation_indices[i][0], validation_indices[i][1], lvl2_validation[i])
                                               for i in range(len(validation_indices))])
    lvl2_validation = utils.clip(lvl2_validation)
    rmse = utils.compute_rmse(ground_truth_ratings, lvl2_validation, validation_indices)
    print("Stacking RMSE:", rmse)

    submission_predictions = utils.ratings_to_matrix(
        [(test_indices[i][0], test_indices[i][1], lvl2_test[i]) for i in range(len(test_indices))])
    submission_predictions = utils.clip(submission_predictions)
    utils.reconstruction_to_predictions(submission_predictions, SUBMISSION_FILE, test_indices)
    print("Stacking submission predictions saved in {}".format(SUBMISSION_FILE))


def bagging(n):
    """
    Implements simple bagging for CF: generate n matrices of same size as data by 
    sampling rows with replacement. Run prediction on each, combine predictions. This
    should improve results since it reduces overall variance.

    :param n:
    :return:
    """
    ranks = [i for i in range(3, 100)]
    regularizations = [0.0005 * i for i in range(400)]
    k = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)

    BAGGING_METHOD = 'svd'
    k = rank = 10
    reg_emb = 0.02
    reg_bias = 0.005

    predictions = []
    sampled_data = np.zeros_like(masked_data)
    sampled_users = np.zeros(masked_data.shape[0])
    for i in range(n):
        for r in range(masked_data.shape[0]):
            random_row = np.random.choice(masked_data.shape[0])
            # keep track of which user (i.e. row) is added. Later, average ratings of duplicates of each user
            sampled_users[r] = random_row
            sampled_data[r, :] = masked_data[random_row, :]
        if BAGGING_METHOD == 'reg_sgd':
            sampled_prediction, _, _ = model_reg_sgd.predict_by_sgd(sampled_data, k, regularization)
        elif BAGGING_METHOD == 'svd':
            imputed_data = np.copy(sampled_data)
            utils.impute_by_avg(imputed_data, True)
            sampled_prediction, _, _ = model_iterated_svd.predict_by_svd(sampled_data, imputed_data, k)
        elif BAGGING_METHOD == 'sf':
            imputed_data = np.copy(sampled_data)
            utils.impute_by_variance(imputed_data)
            u_embeddings, z_embeddings = utils_svd.get_embeddings(imputed_data, rank)
            sampled_prediction, _, _, _, _ = model_sf.predict_by_sf(sampled_data, rank, reg_emb, reg_bias, u_embeddings,
                                                                    z_embeddings)
        prediction = np.zeros_like(masked_data) * np.nan
        for r in range(prediction.shape[0]):
            duplicate_user_predictions = sampled_prediction[np.argwhere(sampled_users == r), :]
            if duplicate_user_predictions.shape[0] > 0:
                prediction[r, :] = np.mean(duplicate_user_predictions, axis=0)
        predictions.append(prediction)

    print("Finished {} runs of bagging...calculating mean of predictions".format(n))
    mean_predictions = compute_mean_predictions(predictions)
    rmse = utils.compute_rmse(data, mean_predictions)
    print("Bagging RMSE:", rmse)

    utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    utils.reconstruction_to_predictions(
        mean_predictions,
        utils.ROOT_DIR + 'data/meta_training_bagging_' + BAGGING_METHOD + '_' +
        datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_first.csv"))
    utils.reconstruction_to_predictions(
        mean_predictions,
        utils.ROOT_DIR + 'data/meta_validation_bagging_' + BAGGING_METHOD + '_' +
        datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=utils.get_validation_indices(utils.ROOT_DIR + "data/validationIndices_second.csv"))
    print("Baggin submission predictions saved in {}".format(SUBMISSION_FILE))


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
    # all_ratings = utils.load_ratings()
    # data = utils.ratings_to_matrix(all_ratings)
    # mean_predictions = compute_mean_predictions(load_predictions_from_files())
    # rmse = utils.compute_rmse(data, mean_predictions)
    # print("mean_predictions rmse:", rmse)
    # utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    # print("Predictions saved in {}".format(SUBMISSION_FILE))

    # bagging(1)

    stacking(load_predictions_from_files('meta_training'), load_predictions_from_files('meta_validation'))


if __name__ == '__main__':
    main()
