#!/bin/bash

module load python/3.6.0
for lambda in 0 0.005 0.001 0.0015 0.002 0.0025
do
  for ((k = 5; k <=100; k += 5))
  do
    bsub "python run.py $k $lambda"
  done
done
