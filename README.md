# CIL: Collaborative filtering
The src folder contains different models to tackle the task of predicting
ratings for the dataset at hand.

The 'src/model_*.py' files can be used via their 'predict_by_*' methods from
clients. In addition, they provide an exemplary usage in their main methods.
Hence, to test a single model with default parameters, you can simply run them
by e.g. `$python src/model_sf.py`. Different models have different optional
parameters.

'src/run_*.py' focus on one particular composition and execution of models.
'mean_predictions.py', 'bagging.py' and 'stacking.py' represent ensembling
methods, making use of individual models.

Note that two different validation principles have been used. For simple model
evaluation, we split the test set in 90% for learning and 10% for validation,
randomized. For ensembling, we used a _three-way-split_, splitting it in 80% for
model learning, 10% for ensemble learning and 10% for validation.

To reproduce the final kaggle submissions, follow the steps listed below. Note that stacked 
ensembling requires two separate runs of training, one on 80% of the data and one on
90% of the data.

1. Run training_validation_split.py where TRAIN_PROPORTION is set to 0.8 and 0.9 and THREE_WAY_SPLIT is set to True and False, respectively. Make sure to place the files in appropriate directories so no files are overwritten. You can also choose to use the files in the repository. The training and validation indices corresponding to a 90% - 10% split are in data/ and the training and validation indices corresponding used for stacking are in data/train_valid_80_10_10/. Make sure to correctly set the paths in utils.py depending on where you save the training data.
2. Run the predictors listed in Table II in the paper accompanying this repository. Specify parameters via the command line, and where no value is mentioned, use the default value specified in the code. Make sure to run them on both the 80% and 90% training sets. You should have a meta_training*, meta_validation* and submission* csv file for each method. 
3. Once all the models have been run put all the meta*.csv files in the locations specified by ENSEMBLE_INPUT_DIR in utils.py. Then run stacking.py, setting STACKING_METHOD to 'nn'. This produces the final submission file. Again, make sure all files are placed in the correct locations. 

Reach out to {hahnb, kklein, kuhnl}@student.ethz.ch if you run into any problems. 


