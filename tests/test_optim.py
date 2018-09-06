from keras.datasets import mnist
from keras.utils import np_utils

from hyperas.optim import retrieve_data_string


def test_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes_return = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes_return)
    Y_test = np_utils.to_categorical(y_test, nb_classes_return)
    return X_train, Y_train, X_test, Y_test


def test_data_function():
    result = retrieve_data_string(test_data, verbose=False)
    assert 'return X_train, Y_train, X_test, Y_test' not in result
    assert 'def data():' not in result
    assert 'nb_classes_return = 10' in result
    assert '(X_train, y_train), (X_test, y_test) = mnist.load_data()' in result
    assert 'Y_test = np_utils.to_categorical(y_test, nb_classes_return)' in result


if __name__ == '__main__':
    test_data_function()
