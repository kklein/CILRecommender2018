import subprocess

for ranks in [7, 10, 15, 20, 25]:
    for n_meta_epochs in [6, 12, 18]:
        args = ["bsub", "-n", "4" "-R", "rusage[mem=8192]", "python", "run_chain.py", str(rank), str(n_meta_epochs)]
        subprocess.Popen(args)
