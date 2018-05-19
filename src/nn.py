import copy
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import NMF
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error

import numpy as np
import utils

DATA_FILE = os.path.join(utils.ROOT_DIR, 'data/data_train.csv')
SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,
                               'data/submission_nn.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/nn_regularization_scores.csv')


def get_embeddings(data, embedding_type, embedding_dimension):
    print("Getting embeddings using {0}".format(embedding_type))
    if embedding_type == "svd":
        u, _, vh = np.linalg.svd(data)
        u = u[:, :embedding_dimension]
        vh = vh[:embedding_dimension, :]
        return u, vh.T
    elif embedding_type == "pca":
        model_1 = PCA(n_components=embedding_dimension)
        w = model_1.fit_transform(data)
        h = model_1.fit_transform(data.T)
        return w, h
    elif embedding_type == "lle":
        model = LocallyLinearEmbedding()
        w = model.fit_transform(data)
        h = model.fit_transform(data.T)
        return w, h
    else:
        if embedding_type == "nmf":
            model = NMF(n_components=embedding_dimension, init='random', random_state=0)
        elif embedding_type == "fa":
            model = FactorAnalysis(n_components=embedding_dimension)

        w = model.fit_transform(data)
        h = model.components_
        return w, h.T


def prepare_data_for_nn(user_embeddings, item_embeddings, data_matrix, n_training_samples):
    """Concatenates user embeddings and item embeddings, and adds corresponding rating from data matrix.
    Returns: x_train, y_train, x_validate, y_validate."""

    x_train = []
    y_train = []

    x_validate = []

    counter = 0
    for i, j in zip(*np.nonzero(data_matrix)):
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        y = data_matrix[i, j]
        x_train.append(x)
        y_train.append(y)
        counter += 1
        if counter > n_training_samples:
            break
        elif counter % 1000 == 0:
            pass
            # print(counter)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_train = np.ravel(y_train)

    indices_to_predict = utils.get_indices_to_predict()

    for i, j in indices_to_predict:
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        x_validate.append(x)

    x_validate = np.asarray(x_validate)

    return x_train, y_train, x_validate


def write_nn_predictions(data_matrix, y_predicted):
    indices_to_predict = utils.get_indices_to_predict()
    for a, index in enumerate(indices_to_predict):
        if a < len(y_predicted):
            data_matrix[index] = y_predicted[a]
        else:
            data_matrix[index] = 3

    utils.reconstruction_to_predictions(data_matrix, SUBMISSION_FILE)


def write_nn_score(score, embedding_type, embedding_dimensions, architecture, n_training_samples, alpha):
    with open(SCORE_FILE, 'a+') as file:
        file.write('{0}, {1}, {2}, {3}, {4}, {5}\n'.format(score, embedding_type, embedding_dimensions, architecture,
                                                      n_training_samples, alpha))


def predict_by_nn(data_matrix, imputed_data, nn_configuration):
    embedding_type, embedding_dimensions, architecture, n_training_samples, alpha = nn_configuration

    # Get embeddings
    print("Getting embeddings of dimension: {0}".format(embedding_dimensions))
    user_embeddings, item_embeddings = get_embeddings(imputed_data, embedding_type, embedding_dimensions)

    x_train, y_train, x_validate = prepare_data_for_nn(user_embeddings, item_embeddings, data_matrix,
                                                       n_training_samples)
    # print(y_train)
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)
    print("Number of training examples: {0}".format(len(x_train)))
    classifier = MLPRegressor(architecture, alpha=alpha)
    print("Classifier parameters {0}".format(classifier.get_params()))
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    print("RMSE: ", rmse)

    write_nn_score(rmse, embedding_type, embedding_dimensions, architecture, n_training_samples * (1 - test_size), alpha)

    y_predicted = classifier.predict(x_validate)
    write_nn_predictions(data_matrix, y_predicted)


def main():
    np.random.seed(10)
    all_ratings = utils.load_ratings()
    data_matrix = utils.ratings_to_matrix(all_ratings)
    imputed_data = utils.predict_by_avg(copy.copy(data_matrix), True)

    if len(sys.argv) == 1:
        embedding_type = "nmf"
        embedding_dimensions = 10
        architecture = (7,)
        n_training_samples = 100000
    else:
        embedding_type = sys.argv[1]
        embedding_dimensions = int(sys.argv[2])
        architecture = eval(sys.argv[3])
        n_training_samples = int(sys.argv[4])
        alpha = float(sys.argv[5])
        print(architecture)

    nn_configuration = (embedding_type, embedding_dimensions, architecture, n_training_samples, alpha)

    predict_by_nn(data_matrix, imputed_data, nn_configuration)


if __name__ == '__main__':
    main()
