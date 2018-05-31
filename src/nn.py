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
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/nn_scores_31.csv')


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
    elif embedding_type == "ids":
        uid_dimensions = len(str(data.shape[0])) - 1
        iid_dimensions = len(str(data.shape[1])) - 1
        w = np.zeros((data.shape[0], uid_dimensions))
        h = np.zeros((data.shape[1], iid_dimensions))

        # for i in range(w.shape[0]):
        #     n = str(i).zfill(uid_dimensions)
        #     for j in range(len(n)):
        #         w[i, j] = n[j]
        #
        # for i in range(h.shape[0]):
        #     n = str(i).zfill(iid_dimensions)
        #     for j in range(len(n)):
        #         h[i, j] = n[j]

        for i in range(w.shape[0]):
            w[i] =  i

        for i in range(h.shape[0]):
            h[i] = i

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
    y_validate = []

    x_test = []

    counter = 0
    validation_indices = utils.get_validation_indices()
    observed_indices = zip(*np.nonzero(data_matrix))
    #train_indices = list(filter(lambda x: x not in validation_indices, observed_indices))
    train_indices = list(set(observed_indices).difference(set(validation_indices)))

    for i, j in train_indices:
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        y = data_matrix[i, j]
        x_train.append(x)
        y_train.append(y)
        counter += 1
        if counter > n_training_samples:
            break
        elif counter % 1000 == 0:
            #print(counter)
            pass

    for i, j in validation_indices:
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        y = data_matrix[i, j]
        x_validate.append(x)
        y_validate.append(y)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_train = np.ravel(y_train)

    x_validate = np.asarray(x_validate)
    y_validate = np.asarray(y_validate)
    y_validate = np.ravel(y_validate)

    indices_to_predict = utils.get_indices_to_predict()

    for i, j in indices_to_predict:
        x = np.concatenate([user_embeddings[i], item_embeddings[j]])
        x_test.append(x)

    x_test = np.asarray(x_test)
    print("x_test", len(x_test))

    return x_train, y_train, x_validate, y_validate, x_test


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


def predict_by_nn(data_matrix, imputed_data, nn_configuration, classifier):
    embedding_type, embedding_dimensions, architecture, n_training_samples, alpha = nn_configuration

    # Get embeddings
    print("Getting embeddings of dimension: {0}".format(embedding_dimensions))
    user_embeddings, item_embeddings = get_embeddings(imputed_data, embedding_type, embedding_dimensions)

    x_train, y_train, x_validate, y_validate, x_test = prepare_data_for_nn(user_embeddings, item_embeddings, data_matrix,
                                                       n_training_samples)

    # print(y_train)
    #test_size = 0.2
    #x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)
    print("Number of training examples: {0}".format(len(x_train)))
    # classifier = MLPRegressor(architecture, alpha=alpha, warm_start=False)
    print("Classifier parameters {0}".format(classifier.get_params()))
    classifier.fit(x_train, y_train)
    y_validate_hat = classifier.predict(x_validate)
    rmse = np.sqrt(mean_squared_error(y_validate, y_validate_hat))
    print("RMSE: ", rmse)

    write_nn_score(rmse, embedding_type, embedding_dimensions, architecture, len(x_train), alpha)

    y_predicted = classifier.predict(x_test)
    print("ypredicted", len(y_predicted))
    # write_nn_predictions(data_matrix, y_predicted)
    return y_predicted

def main():
    np.random.seed(10)
    all_ratings = utils.load_ratings()
    data_matrix = utils.ratings_to_matrix(all_ratings)
    masked_data_matrix = utils.mask_validation(data_matrix)
    # imputed_data = utils.do_smart_init(copy.copy(masked_data_matrix))
    # imputed_data = utils.predict_by_avg(copy.copy(masked_data_matrix), True)
    imputed_data = utils.novel_init(copy.copy(masked_data_matrix))
    print(imputed_data[:10])

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

    if True:
        nn_configuration = (embedding_type, embedding_dimensions, architecture, n_training_samples, alpha)
        classifier =  MLPRegressor(architecture, alpha=alpha, warm_start=False)
        #predict_by_nn(imputed_data, imputed_data, nn_configuration, classifier)
        prediction = predict_by_nn(data_matrix, imputed_data, nn_configuration, classifier)
        write_nn_predictions(data_matrix, prediction)

    if False:
        architecture = (5,)
        nn_configuration = ("svd",10, architecture, n_training_samples, alpha)
        classifier =  MLPRegressor(architecture, alpha=alpha)
        prediction_1 = predict_by_nn(data_matrix, imputed_data, nn_configuration, classifier)
        print(prediction_1[:100])
        architecture = (50,)
        nn_configuration = ("nmf", 100, architecture, n_training_samples, alpha)
        classifier =  MLPRegressor(architecture, alpha=alpha)
        prediction_2 = predict_by_nn(data_matrix, imputed_data, nn_configuration, classifier)
        prediction = np.mean([prediction_1, prediction_2], axis=0)
        print(prediction_2[:100])
        print(prediction[:100])
        write_nn_predictions(data_matrix, prediction)
if __name__ == '__main__':
    main()
