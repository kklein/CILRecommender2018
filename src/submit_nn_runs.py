import subprocess

EMBEDDING_TYPES = ["iterated_svd", "svd", "ids", "nmf", "lle", "pca"]

for n_features in range(10, 200, 10):
    architectures = [
        (int(n_features / 2),),
        (int(n_features / 2), int(n_features / 4)),
        (int(n_features / 2), int(n_features / 4), int(n_features / 8))]
    for embedding_type in EMBEDDING_TYPES:
        for architecture in architectures:
            args = [
                "bsub", "-R", "rusage[mem=6000]", "python", "model_nn.py",
                embedding_type, str(n_features), str(architecture),
                "1500000", "0.0001"]
            subprocess.Popen(args)
