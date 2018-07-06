# CIL: Collaborative filtering
The src folder contains different models to tackle the task of predicting
ratings for the dataset at hand.
The 'src/model_*.py' files can be used via their 'predict_by_*' methods from
clients. In addition, they provide an exemplary usage in their main methods.
Hence, to test a single model with default parameters, you can simply run them
by e.g. `$python src/model_sf.py`. Different models have different optional
parameters.
'src/run_*.py' focus on one particular composition and execution of models.
'mean_predictions.py', 'baggin.py' and 'stacking.py' represent ensembling
methods, making use of individual models.
Note that two different validation principles have been used. For simple model
evaluation, we split the test set in 90% for learning and 10% for validation,
randomized. For ensembling, we used a _three-way-slit_, splitting it in 80% for
model learning, 10% for ensemble learning and 10% for validation.
