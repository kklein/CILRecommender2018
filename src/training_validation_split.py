import numpy as np
import utils

TRAIN_PROPORTION = 0.8
NUM_VALIDATION_SETS = 2


def write_indices_to_file(indices, file_name):
    #file_name = '../data/' + file_name + '.csv'
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
    if NUM_VALIDATION_SETS == 2:
        print("len(validation_indices): {}, np.floor(len(validation_indices) / 2): {}".format(
            len(validation_indices), np.floor(len(validation_indices) / 2)))
        validation_indices_first = validation_indices[:int(np.floor(len(validation_indices) / 2))]
        validation_indices_second = validation_indices[int(np.floor(len(validation_indices) / 2)):]
        write_indices_to_file(validation_indices_first, utils.VALIDATION_FILE_NAME.split(".csv")[0] + "_first.csv")
        write_indices_to_file(validation_indices_second, utils.VALIDATION_FILE_NAME.split(".csv")[0] + "_second.csv")
    else:
        write_indices_to_file(validation_indices, utils.VALIDATION_FILE_NAME)

if __name__ == '__main__':
    main()
