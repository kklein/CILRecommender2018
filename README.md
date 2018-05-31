## How to run stuff in neural collaborative filtering

### Setting up the virtual environment
Execute the following steps on the euler cluster:
1. module load python/2.7.14
2. virtualenv -p python2.7 ncf
3. source ncf/bin/activate
4. pip install -r requirements.txt
5. Tell keras to use Theano as backend by editing ~/.keras/keras.json. Set "backend": "theano".

Refer to the README of neural_collaborative_filtering to see how to run things. 
