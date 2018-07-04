import os
from datetime import datetime

import numpy as np

import utils
import utils_svd
import model_reg_sgd
import model_iterated_svd
import model_sf

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble' + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')
BAGGING_METHOD = 'svd'


def bagging(n):
    """
    Implements simple bagging for CF: generate n matrices of same size as data by 
    sampling rows with replacement. Run prediction on each, combine predictions. This
    should improve results since it reduces overall variance.

    :param n:
    :return:
    """
    reg_emb = 0.02
    reg_bias = 0.005
    ranks = [i for i in range(3, 100)]
    regularizations = [0.0005 * i for i in range(400)]
    k = np.random.choice(ranks)
    rank = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)

    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)

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
    mean_predictions = utils.compute_mean_predictions(predictions)
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


def main():
    bagging(1)


if __name__ == '__main__':
    main()
