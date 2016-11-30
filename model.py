
# Train a deep neural network to drive a car like myself

import pickle

from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def load_data():
    training_file = 'data/train.p'
    testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.33)

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y




def get_model():
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0],
                            kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())

    model.add(Dense(128, input_shape=(32 * 32 * 3,), name="hidden1"))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, name="hidden2"))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(43, name="output"))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    history = model.fit(X_train, y_train,
                        batch_size=126,
                        nb_epoch=3,
                        verbose=1,
                        validation_data=(X_test, y_test))
    model.add(Dense(128, name="hidden2"))

    return model

    # with open('data/test.p', mode='rb') as f:
    test = pickle.load(f)


# X_test = test['features']
# y_test = test['labels']
# X_test = X_test.astype('float32')
# X_test /= 255
# X_test -= 0.5
# Y_test = np_utils.to_categorical(y_test, 43)

# model.evaluate(X_test, Y_test)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=200,
                    nb_epoch=3,
                    verbose=1,
                    validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
