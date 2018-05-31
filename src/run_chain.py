import os
import model_svd as svd
import model_sf as sf
import utils

SUBMISSION_FILE = os.path.join(utils.ROOT_DIR,\
        'data/chain.csv')
SCORE_FILE = os.path.join(utils.ROOT_DIR, 'analysis/chain_scores.csv')

N_META_EPOCHS = 6
N_EPOCHS = 20
REGULARIZATION = [0.02, 0.05]
APPROXIMATION_RANK = 20

def main():
    all_ratings = utils.load_ratings()
    data = utils.ratings_to_matrix(all_ratings)
    masked_data = utils.mask_validation(data)
    imputed_data = utils.novel_init(masked_data)
    for _ in range(N_META_EPOCHS):
        print("Computing embeddings.")
        u_embeddings, z_embeddings =\
                svd.get_embeddings(imputed_data, APPROXIMATION_RANK)
        print("Executings sgd by sf.")
        reconstruction = sf.predict_by_sf(masked_data,
                regularization=REGULARIZATION, n_epochs=N_EPOCHS,
                u_embedding=u_embeddings, z_embedding=z_embeddings)
        # TODO(kkleindev): Figure out whether to use reconstruction or
        # reconstruction used as imputation
        imputed_data = utils.impute(masked_data, reconstruction)
    rsme = utils.compute_rsme(data, imputed_data)
    print(rsme)
    # write_chain_score(SCORE_FILE)
    utils.reconstruction_to_predictions(reconstruction, SUBMISSION_FILE)

if __name__ == '__main__':
    main()
