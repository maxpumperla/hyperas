# Hyperas
A very simple convenience wrapper around hyperopt for fast prototyping with keras models. Hyperas lets you use the power of hyperopt without having to learn the syntax of it. Instead, just define your keras model as you are used to, but use a simple template notation to define hyper-parameter ranges to tune.

## Installation
Hyperas is available on pip, so you can simply use
```{python}
pip install hyperas
```
to install it.

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
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
accuracy = score[1]
return {'loss': -accuracy, 'status': STATUS_OK}
```
The last step is to actually run the optimization, which is done as follows:

```{python}
from hyperas import optim
best_run = optim.minimize(keras_model,
                          algo=tpe.suggest,
                          max_evals=10,
                          trials=Trials())
```
In this example we use at most 10 evaluation runs and the TPE algorithm from hyperopt for optimization.

## Complete example
An extended version of the above example in one script would read as follows:
```{python}
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def keras_model():
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import RMSprop
    from keras.utils import np_utils

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

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1, show_accuracy=True, verbose=2,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    return {'loss': -score[1], 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model,
                              algo=tpe.suggest,
                              max_evals=10,
                              trials=Trials())
```
