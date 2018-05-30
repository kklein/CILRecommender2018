import os
import numpy as np
import utils

# TODO(b-hahn): Define and use meaningful naming scheme.
SUBMISSION_FILE = os.path.join(utils.ROOT_DIR, 'data/ensemble.csv')

def main():
    path = os.path.join(utils.ROOT_DIR, 'data/')
    files = [os.path.join(path, i) for i in os.listdir(path) if \
            os.path.isfile(os.path.join(path,i)) and 'submission_' in i]
    ratings = []
    for file in files:
        file_ratings = utils.load_ratings(file)
        # TODO(b-hahn): Operate on file_ratings in order to manipulate
        # ratings (overall).
        
    reconstruction = np.zeros((utils.USER_COUNT, utils.ITEM_COUNT))
    for row_index, col_index, value in ratings:
        reconstruction[row_index][col_index] = value
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
