import subprocess

for embedding_dimensions in range(10,200,10):
    for alpha in [0.0001, 0.001, 0.01]:
        hidden_layer_width = int(embedding_dimensions / 2)
        args = ["bsub", "-R", "rusage[mem=6000]", "python", "model_nn.py", "svd", str(embedding_dimensions), "("+str(hidden_layer_width)+",)", "1000000", str(alpha)]
        subprocess.Popen(args)
