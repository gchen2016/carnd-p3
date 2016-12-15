# Train a deep neural network to drive a car like myself

import json

import keras
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.layers.core import Lambda
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

from gen_dir_data import ImageDataGen, preprocess_image
from p3_util import MODLE_IMG_WIDTH, \
    MODLE_IMG_HEIGHT


def model_preprocess(img):
    return preprocess_image(img)


def save_model(model, base_name='model'):
    # REF: https://keras.io/getting-started/faq/
    # Save architecture and weights in HDF5 format

    model_file_name = base_name + '.json'
    model_weight_name = base_name + '.h5'

    # Save architecture only in json format
    model_json_str = model.to_json()
    with open(model_file_name, "w") as json_file:
        json.dump(model_json_str, json_file)

    model.save_weights(model_weight_name)

    print("\nModel saved to " + model_file_name + " and " + model_weight_name)


def init_model1(lr=0.0001):
    '''
    Initialize the model1 for training
    :return: the initialized and defined model
    '''
    # input_shape = (160, 320, 3)
    input_shape = (MODLE_IMG_HEIGHT, MODLE_IMG_WIDTH, 3)
    border_mode = 'valid'  # 'valid', 'same'
    # pool_size = (5, 5)
    pool_size = (3, 3)

    print("init_model1")

    model = Sequential()

    ## Input and normalization
    # method not found in drive.py:
    # model.add(Lambda(lambda x: model_preprocess(x),
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=input_shape,
                     output_shape=input_shape))

    ## CNN 1
    model.add(Convolution2D(
        12, 5, 5,
        border_mode=border_mode,
        subsample=(2, 2),
        input_shape=input_shape,
        init='normal'))
    model.add(Activation('relu'))

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
    model.add(MaxPooling2D(pool_size=pool_size))

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

    adamOpt = Adam(lr=lr)
    model.compile(loss='mean_squared_error',
                  optimizer=adamOpt,
                  metrics=['acc'])

    return model


def load_model(model_path):
    print("Loading model " + model_path)
    with open(model_path, 'r') as jfile:
        model = model_from_json(json.load(jfile))
    # model.compile("adam", "mse")
    weights_file = model_path.replace('json', 'h5')
    model.load_weights(weights_file)
    return model


def train3(model_path=None,
           model_save_base_name='model',
           lr=0.001,
           nb_epoch=3,
           data_dirs=[
               'data/sample/'
           ]
           ):
    """
    Use the new data generator.
    Use model1 for training
    Able to use pre-trained model and continue training
    :param model_paths: pre-trained model path
    :param model_save_base_name: the file base name for saving the model
    :param lr: learning rate for training
    :param nb_epoch: number of epoch
    :param data_dirs: training data directories
    :return:
    """

    print("Train3: " + str(model_path) + " lr= " + str(lr) + " epochs= " + str(nb_epoch))
    print("Train3: data_dirs=" + str(data_dirs))
    # nb_epoch = nb_epoch
    if model_path:
        model = load_model(model_path)
        adamOpt = Adam(lr=lr)
        model.compile(loss='mean_squared_error',
                      optimizer=adamOpt,
                      metrics=['acc'])
        model_save_base_name += '_next'
    else:
        model = init_model1(lr)
    model.summary()

    gdata = ImageDataGen(
        data_dirs,
        center_image_only=False,
        # fliplr=True,
        # angle_adjust= 0.2,
        train_size=.95,
        shuffle=True)
    train_num, valid_num = gdata.get_data_sizes()
    print("DataGen train size={0:d} valid size={1:d}".format(
        train_num, valid_num))
    batch_size = 200
    epoch_size = train_num
    # os.system("nvidia-smi")
    data_gen = gdata.gen_data_from_dir(batch_size=batch_size)
    # valid_data = gdata.get_valid_data() ## not good idications for this project

    ## Setup callbacks
    # tensor_board = keras.callbacks.TensorBoard(
    #    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    # cp_path = "ckpt/50hz-"+str(lr)+"-{epoch:02d}-{val_loss:.2f}.h5"
    cp_path = "ckpt/50hz-" + str(lr) + "-{epoch:02d}.h5"
    check_point = keras.callbacks.ModelCheckpoint(
        filepath=cp_path, verbose=1, save_weights_only=True, save_best_only=False)
    # early_stop = keras.callbacks.EarlyStopping(
    #    monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')

    model.fit_generator(
        data_gen,
        samples_per_epoch=epoch_size,
        # samples_per_epoch=1000,
        nb_epoch=nb_epoch,
        # validation_data=valid_data,
        # nb_val_samples=10,
        callbacks=[check_point])

    save_model(model, base_name=model_save_base_name)

def main():
    # train1()
    # train2()
    ### Pre-train
    train3(lr=0.001,
           nb_epoch=3,
           data_dirs=[
               'data/record/50hz/gfull1/'
           ])
    ### Continue train
    """
    train3(model_path='model_1.json',
           lr=0.0001,
           nb_epoch=20,
           data_dirs=[
               'data/record/slow1/'
           ])
    """
if __name__ == '__main__':
    main()
