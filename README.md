# CIL: Collaborative filtering
The src folder contains different models to tackle the task of predicting
ratings for the dataset at hand.
The 'src/model_*.py' files can be used via their 'predict_by_*' methods from
clients. In addition, they provide an exemplary usage in their main methods.
Hence, to test a single model with default parameters, you can simply run them
by e.g. `$python src/model_sf.py`. Different models have different optional
parameters.
The 'src/run_*.py' focus on one particular composition and execution of models.
