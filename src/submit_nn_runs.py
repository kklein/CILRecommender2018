import subprocess

# subprocess.Popen("bsub -R \"rusage[mem=4000]\" python run.py nmf 10 \"(7,)\" 1000000")
for embedding_dimensions in range(10,200,10):
    hidden_layer_width = int(embedding_dimensions / 2)
    args = ["bsub", "-R", "rusage[mem=4000]", "python", "run.py", "nmf", str(embedding_dimensions), "("+str(hidden_layer_width)+",)", "1000000"]
    subprocess.Popen(args)
