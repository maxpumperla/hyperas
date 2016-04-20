# Hyperas [![Build Status](https://travis-ci.org/maxpumperla/hyperas.svg?branch=master)](https://travis-ci.org/maxpumperla/hyperas)  [![PyPI version](https://badge.fury.io/py/hyperas.svg)](https://badge.fury.io/py/hyperas) 
A very simple convenience wrapper around hyperopt for fast prototyping with keras models. Hyperas lets you use the power of hyperopt without having to learn the syntax of it. Instead, just define your keras model as you are used to, but use a simple template notation to define hyper-parameter ranges to tune.

## Installation
```{python}
pip install hyperas
```

## Quick start
Assume you have an existing keras model like the following.
```{python}
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2)
model.add(Dense(10))
model.add(Activation('softmax'))
```
To do hyper-parameter optimization on this model, just wrap the parameters you want to optimize into double curly brackets and choose a distribution over which to run the algorithm. In the above example, let's say we want to optimize for the best dropout probability in both dropout layers. Choosing a uniform distribution over the interval ```[0,1]```, this translates into the following definition.

```{python}
from hyperas.distributions import uniform

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout({{uniform(0, 1)}}))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout({{uniform(0, 1)}}))
model.add(Dense(10))
model.add(Activation('softmax'))
```

After having trained the model, to optimize, we also have to define which evaluation metric of the model is important to us. For example, if we wish to optimize for accuracy, the following example does the trick:

```{python}
score = model.evaluate(X_test, Y_test, verbose=0)
accuracy = score[1]
return {'loss': -accuracy, 'status': STATUS_OK}
```
The last step is to actually run the optimization, which is done as follows:

```{python}
best_run = optim.minimize(model=model,
                          data=data,
                          algo=tpe.suggest,
                          max_evals=10,
                          trials=Trials())
```
In this example we use at most 10 evaluation runs and the TPE algorithm from hyperopt for optimization.

## Complete example
**Note:** It is important to wrap your data and model into functions as shown below, and then pass them as parameters to the minimizer. ```data()``` returns the data the ```model()``` needs. An extended version of the above example in one script reads as follows. This example shows many potential use cases of hyperas, including:
- Varying dropout probabilities, sampling from a uniform distribution
- Different layer output sizes
- Different optimization algorithms to use
- Varying choices of activation functions
- Conditionally adding layers depending on a choice
- Swapping whole sets of layers


```{python}
from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))
        # We can also choose between complete sets of layers
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
```
