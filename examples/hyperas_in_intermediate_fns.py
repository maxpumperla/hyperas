import numpy
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_pairs(x, digit_indices):
    num_classes = 10
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return numpy.array(pairs), numpy.array(labels)

def create_base_network(input_shape,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(dense_filter1, activation='relu')(x)
    x = Dropout(dropout1)(x)
    x = Dense(dense_filter2, activation='relu')(x)
    x = Dropout(dropout2)(x)
    x = Dense(dense_filter3, activation='relu')(x)
    return Model(input, x)

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return numpy.mean(pred == y_true)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def process_data():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    input_shape = x_train.shape[1:]

    # create training+test positive and negative pairs
    digit_indices = [numpy.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [numpy.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    return tr_pairs, tr_y, te_pairs, te_y,input_shape

def data():
    tr_pairs, tr_y, te_pairs, te_y,input_shape = process_data()
    return tr_pairs, tr_y, te_pairs, te_y,input_shape

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_model(tr_pairs, tr_y, te_pairs, te_y,input_shape):
    epochs = 20
    dropout1 = {{uniform(0,1)}}
    dropout2 = {{uniform(0,1)}}
    dense_filter1 = {{choice([64,128,256])}}
    dense_filter2 = {{choice([64,128,256])}}
    dense_filter3 = {{choice([64,128,256])}}
    # network definition
    base_network = create_base_network(input_shape,dense_filter1,dense_filter2,dense_filter3,dropout1,dropout2)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              verbose=1,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    return {'loss': -te_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    tr_pairs, tr_y, te_pairs, te_y,input_shape = data()

    best_run, best_model = optim.minimize(model=create_model, data=data,
    functions = [process_data,create_base_network,euclidean_distance,contrastive_loss,eucl_dist_output_shape,create_pairs,accuracy,compute_accuracy],
    algo=tpe.suggest,max_evals=100,trials=Trials())
    print("best model",best_model)
    print("best run",best_run)
    print("Evalutation of best performing model:")
    loss,te_acc = best_model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y)
    print("best prediction accuracy on test data %0.2f%%" % (100 * te_acc))
