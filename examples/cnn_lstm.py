from __future__ import print_function
from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import uniform, choice
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def data():
    np.random.seed(1337)  # for reproducibility
    max_features = 20000
    maxlen = 100

    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return X_train, X_test, y_train, y_test, maxlen, max_features


def model(X_train, X_test, y_train, y_test, maxlen, max_features):
    embedding_size = 300
    pool_length = 4
    lstm_output_size = 100
    batch_size = 200
    nb_epoch = 1

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout({{uniform(0, 1)}}))
    # Note that we use unnamed parameters here, which is bad style, but is used here
    # to demonstrate that it works. Always prefer named parameters.
    model.add(Convolution1D({{choice([64, 128])}},
                            {{choice([6, 8])}},
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=rand.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print(best_run)
