from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


def data():
    maxlen = 100
    max_features = 20000

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, X_test, y_train, y_test, max_features, maxlen


def model(X_train, X_test, y_train, y_test, max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)

    model.fit(X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              nb_epoch=1,
              validation_split=0.08,
              callbacks=[early_stopping, checkpointer])

    score, acc = model.evaluate(X_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)
