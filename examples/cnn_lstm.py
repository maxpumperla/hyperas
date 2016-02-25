from __future__ import print_function
from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import uniform


def data():
    import numpy as np
    from keras.preprocessing import sequence
    from keras.datasets import imdb

    np.random.seed(1337)  # for reproducibility

    max_features = 20000
    maxlen = 100

    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return X_train, X_test, y_train, y_test, maxlen, max_features


def model(X_train, X_test, y_train, y_test, maxlen, max_features):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM
    from keras.layers.convolutional import Convolution1D, MaxPooling1D

    # Embedding
    embedding_size = 300

    # Convolution
    filter_length = 6
    nb_filter = 64
    pool_length = 4

    # LSTM
    lstm_output_size = 100

    # Training
    batch_size = 60
    nb_epoch = 2

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode='binary')

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
                                show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return {'loss': -score[1], 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(model=model,
                              data=data,
                              algo=rand.suggest,
                              max_evals=5,
                              trials=Trials())
    print(best_run)
