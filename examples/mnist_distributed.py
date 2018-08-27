from hyperas import optim
from hyperas.distributions import quniform, uniform
from hyperopt import STATUS_OK, tpe, mongoexp
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.datasets import mnist
import tempfile


def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Create your model...
    """
    layer_1_size = {{quniform(12, 256, 4)}}
    l1_dropout = {{uniform(0.001, 0.7)}}
    params = {
        'l1_size': layer_1_size,
        'l1_dropout': l1_dropout
    }
    num_classes = 10
    model = Sequential()
    model.add(Dense(int(layer_1_size), activation='relu'))
    model.add(Dropout(l1_dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    out = {
        'loss': -acc,
        'score': score,
        'status': STATUS_OK,
        'model_params': params,
    }
    # optionally store a dump of your model here so you can get it from the database later
    temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
    model.save(temp_name)
    with open(temp_name, 'rb') as infile:
        model_bytes = infile.read()
    out['model_serial'] = model_bytes
    return out


if __name__ == "__main__":
    trials = mongoexp.MongoTrials('mongo://username:pass@mongodb.host:27017/jobs/jobs', exp_key='mnist_test')
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=trials,
                                          keep_temp=True)  # this last bit is important
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
