from random import shuffle
import os
import numpy as np
import utils
import utils_sgd

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/submission_sgd.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/biased15_sgd_scores.csv')
N_EPOCHS = 1
LEARNING_RATE = 0.001
REGULARIZATION = 0.02
EPSILON = 0.0001

def learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        regularization):
    training_indices = utils.get_indeces_from_file(utils.TRAINING_FILE_NAME)
    total_average = np.mean(data[np.nonzero(data)])
    for i in range(n_epochs):
        print("Epoch {0}:".format(i))
        shuffle(training_indices)
        for k, l in training_indices:
            residual = data[k, l] - total_average - u_bias[k] - z_bias[l]\
                    - np.dot(u_embedding[k, :], z_embedding[l, :])
            u_update = LEARNING_RATE * residual * z_embedding[l, :] - \
                    utils.safe_norm(LEARNING_RATE * regularization * \
                    u_embedding[k, :])
            z_update = LEARNING_RATE * residual * u_embedding[k, :] - \
                    utils.safe_norm(LEARNING_RATE * regularization * \
                    z_embedding[l, :])
            u_bias_update = LEARNING_RATE * (residual - regularization *
                    np.absolute(u_bias[k]))
            z_bias_update = LEARNING_RATE * (residual - regularization *
                    np.absolute(z_bias[l]))
            u_embedding[k, :] += u_update
            z_embedding[l, :] += z_update
            u_bias[k] += u_bias_update
            z_bias[l] += z_bias_update

def predict_by_sgd(data, approximation_rank=None, regularization=REGULARIZATION,
        n_epochs=N_EPOCHS, u_embedding=None, z_embedding=None):
    np.random.seed(42)
    if u_embedding is None and z_embedding is None:
        print("Initialize embeddings.")
        u_embedding, z_embedding = utils_sgd.get_initialized_embeddings(
                approximation_rank, data.shape[0], data.shape[1])
    if u_embedding is None or z_embedding is None:
        raise ValueError("embedding is None!")
    u_bias, z_bias = utils_sgd.get_initialized_biases(data)
    # run learn() several times with different data, u and z embeddings
    # to get these datasets, run a function that splits current training data into folds (use sklearn version?)



    # run the meta-learner using the output u/z embeddings as an input to a NN

    # the loss should be the RMSE on the second validation set

    learn(data, u_embedding, z_embedding, u_bias, z_bias, n_epochs,
        regularization)
    total_average = np.mean(data[np.nonzero(data)])
    reconstruction = utils_sgd.reconstruct(u_embedding, z_embedding, total_average, u_bias, z_bias)
    utils.clip(reconstruction)
    return reconstruction

def main():
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
    num_folds = 4
    kfolds_indices = utils.k_folds(masked_data, num_folds)
    # print(kfolds_indices[0][0].shape)
    # print("kfolds_indices:", kfolds_indices)
    for i, indices in enumerate(kfolds_indices):
        # train using num_folds - 1 folds
        training_indices = np.hstack([kfolds_indices[j] for j in range(len(kfolds_indices)) if j != i])
        fold = np.zeros_like(masked_data)
        fold[training_indices] = masked_data[training_indices]
        reconstruction.append(predict_by_sgd(fold, k, regularization))
        validation_indices = [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]

        rmse = utils.compute_fold_rmse(data, reconstruction[-1], validation_indices)
        print("RMSE for fold {}: {}".format(i, rmse))
        utils_sgd.write_sgd_score(rmse, k, regularization, SCORE_FILE)
        utils.reconstruction_to_predictions(reconstruction[-1], SUBMISSION_FILE.split('.csv')[0] + str(i) + '.csv', validation_indices)

#     reconstruction = predict_by_sgd(masked_data, k, regularization)
#     rsme = utils.compute_rsme(data, reconstruction)


if __name__ == '__main__':
    main()
