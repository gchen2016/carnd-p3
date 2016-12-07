# Train a deep neural network to drive a car like myself

import json
import os

import keras
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from gen_dir_data import ImageDataGen
from p3_util import load_data, \
    MODLE_IMG_WIDTH, \
    MODLE_IMG_HEIGHT, \
    init_data_generator, \
    truncated_normal


def save_model(model):
    # REF: https://keras.io/getting-started/faq/
    # Save architecture and weights in HDF5 format

    # Save architecture only in json format
    model_json_str = model.to_json()
    with open("model.json", "w") as json_file:
        json.dump(model_json_str, json_file)

    model.save_weights('model.h5')


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
        12, 5, 5,
        border_mode=border_mode,
        subsample=(2, 2),
        input_shape=input_shape,
        init=truncated_normal))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 2
    model.add(Convolution2D(
        24, 5, 5,
        border_mode=border_mode,
        subsample=(2, 2),
        # init=lambda shape, name: normal(shape, scale=0.01, name=name)),
        init=truncated_normal))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 3
    model.add(Convolution2D(
        36, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1),
        init=truncated_normal))

    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 4
    model.add(Convolution2D(
        64, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1),
        init=truncated_normal))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))

    """
    ## CNN 5
    model.add(Convolution2D(
        64, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1),
        init=truncated_normal))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))
    """

    model.add(Flatten())

    # TODO: Size limitation
    """
    ### Fully Connected
    model.add(Dense(1164, name="hidden1"))
    model.add(Activation('relu'))
    """
    # model.add(Dropout(0.2))
    model.add(Dense(600, init=truncated_normal, name="hidden2"))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(20, init=truncated_normal, name="hidden3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init=truncated_normal, name="output",
                    activation='tanh'))

    adamOpt = Adam(lr=0.00001)
    model.compile(loss='mean_squared_error',
                  optimizer=adamOpt,
                  metrics=['acc'])

    ### setattr(initializations, 'truncated_normal', truncated_normal)

    return model


def init_model1():
    '''
    Initialize the model for training
    :return: the initialized and defined model
    '''
    # input_shape = (160, 320, 3)
    input_shape = (MODLE_IMG_HEIGHT, MODLE_IMG_WIDTH, 3)
    border_mode = 'valid'  # 'valid', 'same'
    # pool_size = (5, 5)
    pool_size = (3, 3)

    print("init_model1")

    ## CNN 1
    model = Sequential()
    model.add(Convolution2D(
        12, 5, 5,
        border_mode=border_mode,
        subsample=(2, 2),
        input_shape=input_shape,
        init='normal'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 2
    model.add(Convolution2D(
        24, 5, 5,
        border_mode=border_mode,
        subsample=(2, 2),
        # init=lambda shape, name: normal(shape, scale=0.01, name=name)),
        init='normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 3
    model.add(Convolution2D(
        36, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1),
        init='normal'))

    model.add(Activation('relu'))
    ## model.add(MaxPooling2D(pool_size=pool_size))

    ## CNN 4
    model.add(Convolution2D(
        64, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1),
        init='normal'))
    model.add(Activation('relu'))
    ## model.add(MaxPooling2D(pool_size=pool_size))

    """
    ## CNN 5
    model.add(Convolution2D(
        64, 3, 3,
        border_mode=border_mode,
        subsample=(1, 1),
        init='normal'))
    model.add(Activation('relu'))
    ### model.add(MaxPooling2D(pool_size=pool_size))
    """

    model.add(Flatten())

    # TODO: Size limitation
    """
    ### Fully Connected
    model.add(Dense(1164, name="hidden1"))
    model.add(Activation('relu'))
    """
    model.add(Dropout(0.5))
    model.add(Dense(600, init='normal', name="hidden2"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, init='normal', name="hidden3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init='normal', name="output",
                    activation='tanh'))

    adamOpt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=adamOpt,
                  metrics=['acc'])

    return model


"""
history = model.fit(X_train, y_train,
                    batch_size=200,
                    nb_epoch=3,
                    verbose=1,
                    validation_data=(X_val, y_val))
"""


# REF: https://keras.io/callbacks/
# loss, accuracy = model.evaluate(X_val, y_val)

def train(model, data_generator,
          valid_generator, y_data, y_valid,
          nb_epoch=10, samples_per_epoch=100):
    """history = model.fit(X, y,
                        batch_size=126,
                        nb_epoch=3,
                        verbose=1,
                        validation_data=(X_valid, y_valid))
                        """
    model.fit_generator(
        data_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch,
        # validation_data=valid_generator,
        # nb_val_samples=50)
    )


def train1():
    X_train, y_train = load_data(data_dir="data/train3/")
    # X_valid, y_valid = load_data(data_dir="data/train2/")
    print(X_train.shape, y_train.shape)
    # test_norm_data(X_train, y_train)
    model = init_model()
    os.system("nvidia-smi")
    model.summary()
    data_gen = init_data_generator(X_train)
    # valid_gen = init_data_generator(X_valid)
    # train(model, data_gen, valid_gen, y_valid, y_valid)
    train(model,
          data_gen.flow(X_train, y_train, batch_size=6),
          None,
          None,
          None)
    save_model(model)


def train2():
    X_train, y_train = load_data(data_dir="data/train3/")
    # X_train, y_train, X_valid, y_valid = stratified_split(
    #    X_train, y_train, train_size=0.1)
    # X_train, y_train, X_valid, y_valid = train_split(
    #   X_train, y_train, train_size=0.3)
    # X_valid, y_valid = load_data(data_dir="data/record/gtrain1/")
    print(X_train.shape, y_train.shape)
    # print(X_valid.shape, y_valid.shape)
    # test_norm_data(X_train, y_train)
    model = init_model()
    # os.system("nvidia-smi")
    model.summary()
    data_gen = init_data_generator(X_train)
    # valid_gen = init_data_generator(X_valid)
    model.fit_generator(
        data_gen.flow(X_train, y_train, batch_size=150),
        samples_per_epoch=len(y_train),
        nb_epoch=3,
        # validation_data=valid_gen.flow(X_valid, y_valid, batch_size=500),
        # nb_val_samples=300)
    )
    save_model(model)


def train3():
    # using the new data generator.
    nb_epoch = 1
    gdata = ImageDataGen(
        data_dirs=[
            'data/record/slow1/',
            'data/record/gtrain1/',
            'data/record/gtrain2/',
            # 'data/record/gtrain3/',
            'data/train1/',
            'data/train2/',
            'data/train3/'],
        center_image_only=False,
        # angle_adjust= 0.2,
        train_size=.8,
        shuffle=True)
    train_num, valid_num = gdata.get_data_sizes()
    print("DataGen train size={0:d} valid size={1:d}".format(
        train_num, valid_num))
    batch_size = 200
    epoch_size = train_num
    ## model = init_model()
    model = init_model1()
    # os.system("nvidia-smi")
    # model.summary()
    data_gen = gdata.gen_data_from_dir(batch_size=batch_size)
    valid_data = gdata.get_valid_data()

    ## Setup callbacks
    # tensor_board = keras.callbacks.TensorBoard(
    #    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    cp_path = "ckpt/model-f6-normal-{epoch:02d}-{val_loss:.2f}.h5"
    check_point = keras.callbacks.ModelCheckpoint(
        filepath=cp_path, verbose=1, save_weights_only=True, save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')

    model.fit_generator(
        data_gen,
        samples_per_epoch=epoch_size,
        nb_epoch=nb_epoch,
        validation_data=valid_data,
        nb_val_samples=batch_size,
        callbacks=[early_stop])

    save_model(model)


def main():
    # train1()
    # train2()
    train3()
    # test_norm_data()
    # X_train, y_train = load_data(data_dir="data/record/ktrain1/")
    # model.summary()
    # input("Press a key to continue...")


if __name__ == '__main__':
    main()
