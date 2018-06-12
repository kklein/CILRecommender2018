#!/bin/bash

module load python_cpu/3.6.0
for ((k = 1; k <=40; k += 1))
do
  bsub -R "rusage[mem=8192]" "python model_svd.py"
done
