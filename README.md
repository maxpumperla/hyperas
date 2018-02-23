# Hyperas [![Build Status](https://travis-ci.org/maxpumperla/hyperas.svg?branch=master)](https://travis-ci.org/maxpumperla/hyperas)  [![PyPI version](https://badge.fury.io/py/hyperas.svg)](https://badge.fury.io/py/hyperas) 
A very simple convenience wrapper around hyperopt for fast prototyping with keras models. Hyperas lets you use the power of hyperopt without having to learn the syntax of it. Instead, just define your keras model as you are used to, but use a simple template notation to define hyper-parameter ranges to tune.

## Installation
```{python}
pip install hyperas
```

## Quick start

Assume you have data generated as such

```{python}
def data():
    x_train = np.zeros(100)
    x_test = np.zeros(100)
    y_train = np.zeros(100)
    y_test = np.zeros(100)
    return x_train, y_train, x_test, y_test
```

and an existing keras model like the following

```{python}
def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2)
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # ... model fitting

    return model
```


To do hyper-parameter optimization on this model,
just wrap the parameters you want to optimize into double curly brackets
and choose a distribution over which to run the algorithm.

In the above example, let's say we want to optimize
for the best dropout probability in both dropout layers.
Choosing a uniform distribution over the interval ```[0,1]```,
this translates into the following definition.
Note that before returning the model, to optimize,
we also have to define which evaluation metric of the model is important to us.
For example, in the following, we optimize for accuracy.


```{python}
from hyperas.distributions import uniform

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # ... model fitting

    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
```

The last step is to actually run the optimization, which is done as follows:

```{python}
best_run = optim.minimize(model=create_model,
                          data=data,
                          algo=tpe.suggest,
                          max_evals=10,
                          trials=Trials())
```
In this example we use at most 10 evaluation runs and the TPE algorithm from hyperopt for optimization.

Check the "complete example" below for more details.


## Complete example
**Note:** It is important to wrap your data and model into functions as shown below, and then pass them as parameters to the minimizer. ```data()``` returns the data the ```create_model()``` needs. An extended version of the above example in one script reads as follows. This example shows many potential use cases of hyperas, including:
- Varying dropout probabilities, sampling from a uniform distribution
- Different layer output sizes
- Different optimization algorithms to use
- Varying choices of activation functions
- Conditionally adding layers depending on a choice
- Swapping whole sets of layers


```{python}
from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
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

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
```

## FAQ

Here is a list of a few popular errors

### `TypeError: require string label`

You're probably trying to execute the model creation code, with the templates, directly in python.
That fails simply because python cannot run the templating in the braces, e.g. `{{uniform..}}`.
The `def create_model(...)` function is in fact not a valid python function anymore.

You need to wrap your code in a `def create_model(...): ...` function,
and then call it from `optim.minimize(model=create_model,...` like in the example.

The reason for this is that hyperas works by doing template replacement
of everything in the `{{...}}` into a separate temporary file,
and then running the model with the replaced braces (think jinja templating).

This is the basis of how hyperas simplifies usage of hyperopt by being a "very simple wrapper".


### `TypeError: 'generator' object is not subscriptable`

This is currently a [known issue](https://github.com/maxpumperla/hyperas/issues/125).

Just `pip install networkx==1.11`


### `NameError: global name 'X_train' is not defined`

Maybe you forgot to return the `x_train` argument in the `def create_model(x_train...)` call
from the `def data(): ...` function.

You are not restricted to the same list of arguments as in the example.
Any arguments you return from `data()` will be passed to `create_model()`

### notebook adjustment

If you find error like ["No such file or directory"](https://github.com/maxpumperla/hyperas/issues/83) or [OSError, Err22](https://github.com/maxpumperla/hyperas/issues/149), you may need add `notebook_name='simple_notebook'`(assume your current notebook name is `simple_notebook`) in `optim.minimize` function like this:

```python
best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials(),
                                      notebook_name='simple_notebook')
```

