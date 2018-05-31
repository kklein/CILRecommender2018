import os
import numpy as np
import utils

# This file is meant as a template. Feel free to change, replace or copy.
# TODO(b-hahn): Define and use meaningful naming scheme.
SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble.csv')

def main():
    path = os.path.join(utils.ROOT_DIR, 'data/')
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path,i)) and 'submission_' in i]
    reconstruction = np.zeros((utils.USER_COUNT, utils.ITEM_COUNT),
            dtype=np.float64)
    for file in files:
        print('file: ' + str(file))
        file_ratings = utils.load_ratings(file)
        # TODO(b-hahn): Operate on file_ratings in order to manipulate
        # ratings (overall).
        for i,j,v in file_ratings:
            reconstruction[i][j] += v
    for i, j, v in file_ratings:
        reconstruction[i][j] = reconstruction[i][j] / len(files)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
