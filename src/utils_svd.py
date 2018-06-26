import numpy as np

def write_svd_score(score, k, take_bias, score_file):
    with open(score_file, 'a+') as file:
        file.write('%d, %s, %f\n' % (k, str(take_bias), score))

def get_embeddings(imputed_data, approximation_rank):
    u_embeddings, singular_values, z_embeddings =\
            np.linalg.svd(imputed_data)
    u_embeddings = u_embeddings[:, 0:approximation_rank]
    z_embeddings = z_embeddings[0:approximation_rank, :]
    singular_values = singular_values[:approximation_rank]
    singular_values_sqrt = np.zeros((approximation_rank, approximation_rank))
    np.fill_diagonal(singular_values_sqrt, np.sqrt(singular_values))
    u_embeddings = np.matmul(u_embeddings, singular_values_sqrt)
    z_embeddings = np.matmul(singular_values_sqrt, z_embeddings)
    z_embeddings = z_embeddings.T
    return u_embeddings, z_embeddings
