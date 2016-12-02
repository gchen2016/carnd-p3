
# Train a deep neural network to drive a car like myself

import csv
import os

import matplotlib.pyplot as pyplot
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img


# from scipy.misc import toimage


def load_data(dir='data/sample/'):
    # print(rcsetup.all_backends)
    pyplot.ion()
    cvs_file_name = dir + '/driving_log.csv'
    img_dir = 'IMG/'
    img_width = 160;
    img_height = 320;

    data_size = sum(1 for line in open(cvs_file_name))
    X_train = np.zeros((data_size, img_width, img_height, 3))
    y = np.zeros(data_size)

    with open(cvs_file_name, mode='r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        for (line, i) in zip(readcsv, range(data_size)):
            # center image file name
            img_file = dir + img_dir + os.path.basename(line[0])
            # print(img_file, line[3])
            img = load_img(img_file)
            X_train[i, :, :, :] = img_to_array(img)
            # center image driving angle
            y[i] = float(line[3])
            # Show the images
            # if i < 8:
            #  pyplot.subplot(331 + i)
            #  pyplot.imshow(img)
    # pyplot.subplot(339)
    # pyplot.plot(y)
    # pyplot.show()
    # print(X_train.shape, y.shape)
    return X_train, y


def init_data_generator(data):
    # REF: https://keras.io/preprocessing/image/
    datagen = ImageDataGenerator(
        samplewise_center=True,  # Set each sample mean to 0.
        samplewise_std_normalization=True  # Divide each input by its std.
    )
    datagen.fit(data)
    return datagen


# def getNextBatch(datagenerator, data, y_data, batch_size=16):
#    # return batch_X, batch_y
#    batch_X, batch_y = datagenerator.flow(data, y_data, batch_size)
#    return batch_X, batch_y

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


def save_model(model):
    # REF: https://keras.io/getting-started/faq/
    # Save architecture and weights in HDF5 format
    model.save('model.h5')

    # Save architecture only in json format
    json_string = model.to_json()

    # Save weights
    model.save_wights('model_weights.h5')


def init_model():
    '''
    Initialize the model for training
    :return: the initialized and defined model
    '''
    input_shape = (3, 256, 256)
    border_mode = 'valid'  # or  'same'
    nb_filters = 64  # num of output filters
    kernel_size = [5, 5]  # Kernel size
    subsample = (1, 1)  # strides

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0],
                            kernel_size[1],
                            border_mode='valid',
                            subsample=subsample,
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

# model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
"""
history = model.fit(X_train, y_train,
                    batch_size=200,
                    nb_epoch=3,
                    verbose=1,
                    validation_data=(X_val, y_val))
"""


# REF: https://keras.io/callbacks/
# loss, accuracy = model.evaluate(X_val, y_val)

def test_norm_data(X, y):
    pyplot.axis("off")
    pyplot.subplot(521)
    pyplot.imshow(array_to_img(X[0]))
    pyplot.subplot(522)
    pyplot.hist(X[0][0])
    datagen = init_data_generator(X)
    # data_iter = datagen.flow(X, y, batch_size=3, save_to_dir="cache")
    b_size = 3
    data_iter = datagen.flow(X, y, batch_size=b_size)
    c = 0
    for i in data_iter:
        img_a = i[0][c]
        assert img_a.shape[-1] == 3, "Not 3 channels"
        print(img_a.shape)
        # print (img_a)

        pyplot.subplot(523 + c * 2)
        pyplot.imshow(array_to_img(img_a))
        pyplot.subplot(524 + c * 2)
        pyplot.hist(img_a[1])
        c += 1
        if c >= b_size:
            break
    pyplot.show()

def main():
    X_train, y = load_data()
    # print(X_train[1], y)
    test_norm_data(X_train, y)
    input("Press a key to continue...")


if __name__ == '__main__':
    main()
