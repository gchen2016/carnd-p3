# Train a deep neural network to drive a car like myself

import csv
import os

import matplotlib.pyplot as pyplot
import numpy as np
import scipy
from keras.layers import Convolution2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img

### Golbal var
MODLE_IMG_WIDTH = 86
MODLE_IMG_HEIGHT = 43

def load_data(data_dir='data/sample/'):
    # print(rcsetup.all_backends)
    pyplot.ion()
    cvs_file_name = data_dir + '/driving_log.csv'
    img_dir = 'IMG/'
    img_width = MODLE_IMG_WIDTH;
    img_height = MODLE_IMG_HEIGHT;

    data_size = sum(1 for line in open(cvs_file_name))
    X = np.zeros((data_size, img_height, img_width, 3))
    y = np.zeros(data_size)

    with open(cvs_file_name, mode='r') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        for (line, i) in zip(readcsv, range(data_size)):
            # center image file name
            img_file = data_dir + img_dir + os.path.basename(line[0])
            # print(img_file, line[3])
            img = load_img(img_file)  # orginal size is 320 x 160
            img = scipy.misc.imresize(img, (img_height, img_width))
            X[i, :, :, :] = img_to_array(img)
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
    return X, y


def init_data_generator(data):
    # REF: https://keras.io/preprocessing/image/
    datagen = ImageDataGenerator(
        samplewise_center=True,  # Set each sample mean to 0.
        samplewise_std_normalization=True  # Divide each input by its std.
    )
    datagen.fit(data)
    return datagen

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
    # input_shape = (160, 320, 3)
    input_shape = (MODLE_IMG_HEIGHT, MODLE_IMG_WIDTH, 3)
    border_mode = 'valid'  # 'valid', 'same'
    pool_size = (5, 5)

    ## CNN 1
    model = Sequential()
    model.add(Convolution2D(
        24, 5, 5,
        border_mode=border_mode,
        subsample=(1, 1),
        input_shape=input_shape))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 2
    model.add(Convolution2D(
        36, 5, 5,
        border_mode=border_mode,
        subsample=(1, 1)))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 3
    model.add(Convolution2D(
        48, 5, 5,
        border_mode=border_mode,
        subsample=(1, 1)))

    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 4
    model.add(Convolution2D(
        64, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1)))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 5
    model.add(Convolution2D(
        64, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1)))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))


    model.add(Flatten())

    ### Fully Connected
    model.add(Dense(1164, name="hidden1"))
    model.add(Activation('relu'))

    # model.add(Dropout(0.2))
    model.add(Dense(150, name="hidden2"))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(10, name="hidden3"))
    model.add(Activation('relu'))
    model.add(Dense(1, name="output"))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['acc'])

    return model


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
        if c >= b_size - 1:
            break

    # show y data
    pyplot.subplot(527)
    pyplot.plot(y)
    pyplot.subplot(528)
    pyplot.hist(y)
    pyplot.show()

def train(model, data_generator, valid_generator, y_data, y_valid):
    """history = model.fit(X, y,
                        batch_size=126,
                        nb_epoch=3,
                        verbose=1,
                        validation_data=(X_valid, y_valid))
                        """
    model.fit_generator(
        data_generator,
        samples_per_epoch=10,
        nb_epoch=100,
        # validation_data=valid_generator,
        # nb_val_samples=50)
    )

def main():
    X_train, y_train = load_data(data_dir="data/train1/")
    X_valid, y_valid = load_data(data_dir="data/train2/")
    # print(X_train[1], y)
    test_norm_data(X_train, y_train)
    model = init_model()
    print(model.to_json())
    data_gen = init_data_generator(X_train)
    # valid_gen = init_data_generator(X_valid)
    # train(model, data_gen, valid_gen, y_valid, y_valid)
    train(model,
          data_gen.flow(X_train, y_train, batch_size=32),
          None,
          y_train,
          y_valid)

    input("Press a key to continue...")


if __name__ == '__main__':
    main()
