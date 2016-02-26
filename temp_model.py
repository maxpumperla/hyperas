from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb

np.random.seed(1337)  # for reproducibility

max_features = 20000
maxlen = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)





def keras_fmin_fnct(space):

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
    batch_size = 200
    nb_epoch = 1

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(space['Dropout']))
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
    return {'loss': -acc, 'status': STATUS_OK}


def get_space():
 return {
'Dropout': hp.uniform('Dropout', 0, 1)
}
