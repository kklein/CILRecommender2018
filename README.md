# CIL: Collaborative filtering
The src folder contains different models to tackle the task of predicting
ratings for the dataset at hand.
The 'src/model_*.py' files can be used via their 'predict_by_*' methods from
clients. In addition, they provide an exemplary usage in their main methods.
Hence, to test a single model with default parameters, you can simply run them
by e.g. `$python src/model_sf.py`. Different models have different optional
parameters.
The 'src/run_*.py' focus on one particular composition and execution of models.

## How to run stuff in neural collaborative filtering

### Setting up the virtual environment
Execute the following steps on the euler cluster:
1. module load python/2.7.14
2. virtualenv -p python2.7 ncf
3. source ncf/bin/activate
4. pip install -r requirements.txt
5. Tell keras to use Theano as backend by editing ~/.keras/keras.json. Set "backend": "theano".

Refer to the README of neural_collaborative_filtering to see how to run things.
You might run `python NeuMF.py --dataset cil --epochs 20 --batch_size 256 --num_factors
 8 --layers [10] --reg_mf 0 --reg_layers [0] --num_neg 0 --lr 0.001 --learner adam --verbose 1 --out 1` for instance. Submissions are stored in `neural_collaborative_filtering/Data`.
