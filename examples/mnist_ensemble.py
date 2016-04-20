from __future__ import print_function
from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop


def data():
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test


def model(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([400, 512, 600])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

    nb_epoch = 10
    batch_size = 128

    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=2,
              validation_data=(X_test, Y_test))

    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = data()

    '''
    Generate ensemble model from optimization run:
    First, run hyperas optimization on specified setup, i.e. 10 trials with TPE,
    then return the best 5 models and create a majority voting model from it.
    '''
    ensemble_model = optim.best_ensemble(nb_ensemble_models=5,
                                         model=model, data=data,
                                         algo=rand.suggest, max_evals=10,
                                         trials=Trials(),
                                         voting='hard')
    preds = ensemble_model.predict(X_test)
    y_test = np_utils.categorical_probas_to_classes(Y_test)
    print(accuracy_score(preds, y_test))
