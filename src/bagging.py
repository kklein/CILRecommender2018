import os
from datetime import datetime
import numpy as np
import utils
import utils_svd
import model_reg_sgd
import model_iterated_svd
import model_sf

SUBMISSION_FILE = os.path.join(
    utils.ROOT_DIR, 'data/ensemble' +
    datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')
SF_REG_EMB = 0.02
SF_REG_BIAS = 0.05
N_MATRICES = 30

def bagging(n_matrices, bagging_method, data, masked_data, rank,
            regularization):
    """
    Implements simple bagging for CF: generate n_matric matrices of same
    size as data by sampling rows with replacement. Run prediction on each,
    combine predictions. This should improve results since it reduces overall
    variance.

    :param n_matrices:
    :return:
    """
    predictions = []
    sampled_data = np.zeros_like(masked_data)
    sampled_users = np.zeros(masked_data.shape[0])
    for _ in range(n_matrices):
        for new_row_index in range(masked_data.shape[0]):
            random_row = np.random.choice(masked_data.shape[0])
            # Keep track of which user (i.e. row) is added. Later, average
            # ratings of duplicates of each user.
            sampled_users[new_row_index] = random_row
            sampled_data[new_row_index, :] = masked_data[random_row, :]
        if bagging_method == 'reg_sgd':
            sampled_prediction, _, _ = model_reg_sgd.predict_by_sgd(
                sampled_data, rank, regularization)
        elif bagging_method == 'svd':
            imputed_data = np.copy(sampled_data)
            utils.impute_by_avg(imputed_data, True)
            sampled_prediction, _, _ = model_iterated_svd.predict_by_svd(
                sampled_data, imputed_data, rank)
        elif bagging_method == 'sf':
            imputed_data = np.copy(sampled_data)
            utils.impute_by_variance(imputed_data)
            u_embeddings, z_embeddings = utils_svd.get_embeddings(
                imputed_data, rank)
            sampled_prediction, _, _, _, _ = model_sf.predict_by_sf(
                sampled_data, rank, SF_REG_EMB, SF_REG_BIAS, u_embeddings,
                z_embeddings)
        prediction = np.zeros_like(masked_data) * np.nan
        for row_index in range(prediction.shape[0]):
            duplicate_user_predictions = sampled_prediction[
                np.argwhere(sampled_users == row_index), :]
            if duplicate_user_predictions.shape[0] > 0:
                prediction[row_index, :] = np.mean(
                    duplicate_user_predictions, axis=0)
        predictions.append(prediction)

    print("Finished {} runs of bagging...calculating mean of\
          predictions".format(n_matrices))
    mean_predictions = utils.compute_mean_predictions(predictions)
    rmse = utils.compute_rmse(data, mean_predictions)
    print("Bagging RMSE:", rmse)

    utils.reconstruction_to_predictions(mean_predictions, SUBMISSION_FILE)
    utils.reconstruction_to_predictions(
        mean_predictions,
        utils.ROOT_DIR + 'data/meta_training_bagging_' + bagging_method + '_' +
        datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=utils.get_validation_indices(True))
    utils.reconstruction_to_predictions(
        mean_predictions,
        utils.ROOT_DIR + 'data/meta_validation_bagging_' + bagging_method + '_'
        + datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv',
        indices_to_predict=utils.get_meta_validation_indices())
    print("Baggin submission predictions saved in {}".format(SUBMISSION_FILE))

def main():
    bagging_method = 'svd'
    ranks = [i for i in range(3, 100)]
    regularizations = [0.0005 * i for i in range(400)]
    rank = np.random.choice(ranks)
    regularization = np.random.choice(regularizations)
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data, True)
    bagging(N_MATRICES, bagging_method, data, masked_data, rank, regularization)

if __name__ == '__main__':
    main()
