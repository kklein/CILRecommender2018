import numpy as np
import utils

TRAIN_PROPORTION = 0.9

def write_indices_to_file(indices, file_name):
    file_name = '../data/' + file_name + '.csv'
    with open(file_name, 'w') as file:
        file.write('Row,Column\n')
        for i, j in indices:
            file.write('%d, %d\n' % (i, j))

def main():
    all_ratings = utils.load_ratings('../data/data_train.csv')
    data_matrix = utils.ratings_to_matrix(all_ratings)
    observed_indices = utils.get_observed_indeces(data_matrix)
    np.random.shuffle(observed_indices)
    split_threshold = round(TRAIN_PROPORTION * len(observed_indices))
    training_indices = observed_indices[:split_threshold]
    validation_indices = observed_indices[split_threshold:]
    training_indices.sort(key=lambda x: (x[0], x[1]))
    validation_indices.sort(key=lambda x: (x[0], x[1]))
    write_indices_to_file(training_indices, utils.TRAINING_FILE_NAME)
    write_indices_to_file(validation_indices, utils.VALIDATION_FILE_NAME)

if __name__ == '__main__':
    main()
