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
ENSEMBLE_INPUT_DIR = 'data/stacking/90/'
STACKING_METHOD = 'lr'


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


def main():
    stacking(load_predictions_from_files('meta_training'), load_predictions_from_files('meta_validation'))


if __name__ == '__main__':
    main()
