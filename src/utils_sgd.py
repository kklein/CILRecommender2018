import numpy as np
import utils

def write_sgd_score(score, approximation_rank, reg_emb, reg_bias, smoothed,
                    initialization_string, score_file):
    with open(score_file, 'a+') as file:
        file.write('%d, %f, %f, %s, %s, %f\n' % (
            approximation_rank, reg_emb, reg_bias, smoothed,
            initialization_string, score))

def get_initialized_biases(data):
    training_indices = utils.get_indices_from_file(utils.TRAINING_FILE_NAME)
    u_bias = np.zeros(data.shape[0], dtype=np.float128)
    u_counters = np.zeros(data.shape[0], dtype=np.float128)
    z_bias = np.zeros(data.shape[1], dtype=np.float128)
    z_counters = np.zeros(data.shape[1], dtype=np.float128)
    total_average = np.mean(data[np.nonzero(data)])
    for k, l in training_indices:
        u_bias[k] += data[k][l]
        u_counters[k] += 1
        z_bias[l] += data[k][l]
        z_counters[l] += 1
    # 25 is a constant that works 'well' empirically.
    # http://sifter.org/simon/journal/20061211.html
    for k in range(data.shape[0]):
        u_bias[k] = ((u_bias[k] + 25 * total_average) / (25 + u_counters[k]))\
                - total_average
    for l in range(data.shape[1]):
        z_bias[l] = ((z_bias[l] + 25 * total_average) / (25 + z_counters[l]))\
                - total_average
    return u_bias, z_bias

def get_initialized_embeddings(approximation_rank, u_rows, z_rows):
    if approximation_rank is None:
        raise ValueError("Approximation rank has not been given.")
    u_embeddings = np.random.rand(u_rows, approximation_rank).astype(
        np.float128)
    z_embeddings = np.random.rand(z_rows, approximation_rank).astype(
        np.float128)
    return u_embeddings, z_embeddings

def reconstruct(u_embeddings, z_embeddings, u_bias, z_bias, total_average=0):
    # Reshape arrays in order to allow for broadcasting on matrix.
    u_bias = np.reshape(u_bias, (u_bias.shape[0], 1))
    z_bias = np.reshape(z_bias, (1, z_bias.shape[0]))
    prod = np.dot(u_embeddings, z_embeddings.T)
    prod += total_average
    prod += u_bias
    prod += z_bias
    return prod
