import os
from datetime import datetime

import numpy as np
from sklearn import ensemble

import utils

# This file is meant as a template. Feel free to change, replace or copy.
# TODO(b-hahn): Define and use meaningful naming scheme.
SUBMISSION_FILE = os.path.join(
    utils.ROOT_DIR, 'data/ensembles/ensemble' +
    datetime.now().strftime('%Y-%b-%d-%H-%M-%S') + '.csv')


def main():
    path = os.path.join(utils.ROOT_DIR, 'data/ensembles')
    # TODO: instead of using all submissions, calculate their respective
    # correlations and use only the least correlated.
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path,i)) and 'submission_' in i]
    reconstruction = np.zeros((utils.USER_COUNT, utils.ITEM_COUNT),
            dtype=np.float64)
    ratings = []
    all_ratings = []
    for n, file in enumerate(files):
        print("loading {}".format(file))
        # file_ratings = utils.load_ratings(file)
        all_ratings.append(utils.load_ratings(file))

    # get majority vote on each rating.
    # TODO: need to deal with ties in some way?
    num_ratings = len(all_ratings[0])
    for i in range(num_ratings):
        # TODO: remove int() to get mean. Should always perform worse than voting, right?
        # TODO: weight submissions according their validation score (i.e higher weight/
        # more votes for better performing models)
        # majority_rating = int(np.round(np.mean([r[i][2] for r in all_ratings])))
        majority_rating = np.mean([r[i][2] for r in all_ratings])
        # print([r[i] for r in all_ratings])
        # print(majority_rating)
        reconstruction[all_ratings[0][i][0], all_ratings[0][i][1]] = majority_rating
        #print(all_ratings[0][i][0], all_ratings[0][i][1])

        # ratings.append((all_ratings[0][i][0], all_ratings[0][i][1],
        #                 majority_rating))

    # for i, j, value in ratings:
    #     reconstruction[i][j] = value
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)


if __name__ == '__main__':
    main()
