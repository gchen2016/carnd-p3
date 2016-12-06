import csv
import os
import random

import numpy as np
from keras.preprocessing.image import \
    load_img, \
    img_to_array


class ImageDataGen(object):
    '''
    Generate image batches when needs to save mem
    '''

    def __init__(self,
                 data_dir="data/sample/",
                 data_size=None,
                 label_only=False,
                 center_image_only=True,
                 angle_adjust=0.5,
                 train_size=0.8,
                 batch_size=60):

        self.gen_train_batch_index = 0
        self.gen_train_batch_size = batch_size
        self.gen_valid_batch_index = 0
        self.gen_loaded_size = 0
        self.xnames = []  # all image file names
        self.yangles = []  # all steering angles
        self.train_num = 0
        self.valid_num = 0
        self.x_train_names = []
        self.y_train_angles = []
        self.x_valid_names = []
        self.y_valid_angles = []

        self.csv_name = 'driving_log.csv'
        self.img_dir = 'IMG/'
        self.data_dir = data_dir
        self.label_only = label_only
        self.center_image_only = center_image_only
        self.angle_adjust = angle_adjust
        self.train_size = train_size

        csv_file_name = self.data_dir + self.csv_name
        # img_width = MODLE_IMG_WIDTH;
        # img_height = MODLE_IMG_HEIGHT;

        # load image names and angle
        with open(csv_file_name, mode='r') as csvfile:
            readcsv = csv.reader(csvfile, delimiter=',')
            for line in readcsv:
                # center image file name
                img_file = self.data_dir + self.img_dir + os.path.basename(line[0])
                self.xnames.append(img_file)
                self.yangles.append(float(line[3]))
                # left image
                if not self.center_image_only:
                    img_file = self.data_dir + self.img_dir + os.path.basename(line[1])
                    self.xnames.append(img_file)
                    self.yangles.append(float(line[3]) + self.angle_adjust)
                    # right image
                    img_file = self.data_dir + self.img_dir + os.path.basename(line[2])
                    self.xnames.append(img_file)
                    self.yangles.append(float(line[3]) - self.angle_adjust)

        num_angles = len(self.yangles)
        # shuffle list of angles
        l = list(zip(self.xnames, self.yangles))
        random.shuffle(l)
        self.xnames, self.yangles = zip(*l)
        self.train_num = int(len(self.xnames) * self.train_size)
        self.x_train_names = self.xnames[:self.train_num]
        self.y_train_angles = self.yangles[:self.train_num]
        self.x_valid_names = self.xnames[-self.train_num:]
        self.y_valid_angles = self.yangles[-self.train_num:]
        self.valid_num = len(self.y_valid_angles)

    def get_train_data(self):
        X = [img_to_array(load_img(i)) / 255. for i in self.x_train_names]
        return X, self.y_train_angles

    def get_valid_data(self):
        X = [img_to_array(load_img(i)) / 255. for i in self.x_valid_names]
        return X, self.y_valid_angles

    def gen_data_from_dir(self):
        train_index = 0
        while 1:
            if train_index + self.gen_train_batch_size > self.train_num
                train_index = 0
                paths = self.x_train_names[train_index: train_index + self.gen_train_batch_size]
                X = [img_to_array(load_img(i)) / 255. for i in paths]
                y = self.y_train_angles[train_index: train_index + self.gen_train_batch_size]
                train_index += self.gen_train_batch_size
                yield np.array(X), np.array(y)
