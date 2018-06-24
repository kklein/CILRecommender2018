import subprocess

for embedding_dimensions in range(10,200,10):
    for embedding_type in ["iterated_svd", "svd", "ids", "nmf", "lle", "pca"]:
        hidden_layer_width = int(embedding_dimensions / 2)
        for architecture in [(int(embedding_dimensions / 2),), (int(embedding_dimensions / 2), int(embedding_dimensions / 4)), (int(embedding_dimensions / 2),int(embedding_dimensions / 4),int(embedding_dimensions / 8))]:

            args = ["bsub", "-R", "rusage[mem=6000]", "python", "model_nn.py", embedding_type, str(embedding_dimensions), str(architecture), "1500000","0.0001"]
            subprocess.Popen(args)
