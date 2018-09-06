from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def data():
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
    model = Sequential()
    model.add(Dense(44, input_shape=(784,)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense(44))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense(10))

    model.compile(loss='mae', metrics=['mse'], optimizer="adam")

    es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10)
    rlr = ReduceLROnPlateau(factor=0.1, patience=10)
    _ = model.fit(x_train, y_train, epochs=1, verbose=0, callbacks=[es, rlr],
                  batch_size=24, validation_data=(x_test, y_test))

    mae, mse = model.evaluate(x_test, y_test, verbose=0)
    print('MAE:', mae)
    return {'loss': mae, 'status': STATUS_OK, 'model': model}


def test_advanced_callbacks():
    X_train, Y_train, X_test, Y_test = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials(),
                                          verbose=False)
