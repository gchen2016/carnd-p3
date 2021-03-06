import csv
import os
import random

import numpy as np
from keras.preprocessing.image import \
    load_img, \
    img_to_array

IMG_WIDTH = 320
IMG_HEIGHT = 160

def preprocess_image(img):
    """
    normalize the image data
    :param img: The image to be normalized
    :return: The normalized image
    """
    # return img/255.  # normalize to (0, - 1)
    return img / 127.5 - 1.  # normalize to (-1, 1)

def load_processed_image(img_path):
    """
    :param img_path: The path includes the preprocessing flags like :FLR
    :return: The loaded and preprocessed image
    """
    p = img_path.split(':')  # extract the path and flags
    img = img_to_array(load_img(p[0]))
    if len(p) > 1:
        if p[1] == 'FLR':  # fliplr flag
            img = np.fliplr(img)
    # Lambda did this: return preprocess_image(img)
    return img

class ImageDataGen(object):
    """
    Generate image batches when needs to save mem
    """

    def __init__(self,
                 data_dirs=["data/sample/"],
                 label_only=False,
                 center_image_only=True,
                 fliplr=True,
                 shuffle=True,
                 angle_adjust=0.5,
                 train_size=0.8,
                 batch_size=120):
        """
        :param data_dirs: A list of directory to load the data
        :param label_only: Load the label only for debugging and testing
        :param center_image_only: If True, load the center images only
        :param fliplr: If True, add left-right fliped images for augmentation
        :param shuffle: If True, to shuffle the images.
        :param angle_factor: How much to adjust the left and right steering angle toward the center.
        """

        self.img_width = IMG_WIDTH
        self.img_height = IMG_HEIGHT
        self.img_channels = 3
        self.batch_index = 0
        self.batch_size = batch_size
        self.xnames = []  # all image file names
        self.yangles = []  # all steering angles
        self.data_num = 0
        self.train_num = 0
        self.valid_num = 0
        self.x_train_names = []
        self.y_train_angles = []
        self.x_valid_names = []
        self.y_valid_angles = []
        self.csv_name = 'driving_log.csv'
        self.img_dir = 'IMG/'
        self.data_dirs = data_dirs
        self.label_only = label_only
        self.center_image_only = center_image_only
        self.shuffle = shuffle
        self.angle_adjust = angle_adjust
        self.train_size = train_size
        self.angle_factor = 0.75

        for d in self.data_dirs:
            csv_file_name = d + self.csv_name
            # img_width = MODLE_IMG_WIDTH;
            # img_height = MODLE_IMG_HEIGHT;

            # load image names and angle
            with open(csv_file_name, mode='r') as csvfile:
                readcsv = csv.reader(csvfile, delimiter=',')
                for line in readcsv:
                    # center image file name
                    center_angle = float(line[3])
                    """
                    if abs(center_angle) > angle_clip:
                        ## continue # bad labels
                        center_angle *= angle_clip
                        print(csv_file_name, center_angle)
                    """
                    img_file = d + self.img_dir + os.path.basename(line[0])
                    self.xnames.append(img_file)
                    self.yangles.append(center_angle)
                    if fliplr:
                        self.xnames.append(img_file + ":FLR")  # add keyword for augment
                        self.yangles.append(center_angle * -1.)
                    # left image
                    if not self.center_image_only:
                        img_file = d + self.img_dir + os.path.basename(line[1])
                        self.xnames.append(img_file)
                        if center_angle < 0.:  ## left turn
                            self.yangles.append(
                                center_angle +
                                abs(center_angle * self.angle_factor))
                        else:
                            self.yangles.append(
                                center_angle + abs(center_angle * self.angle_factor))
                        # right image
                        img_file = d + self.img_dir + os.path.basename(line[2])
                        self.xnames.append(img_file)
                        if center_angle > 0:
                            self.yangles.append(
                                center_angle -
                                abs(center_angle * self.angle_factor))
                        else:
                            self.yangles.append(
                                center_angle -
                                abs(center_angle * self.angle_factor))

        self.check_and_shuffle_index()
        self.data_num = len(self.xnames)
        self.train_num = int(self.data_num * self.train_size)
        self.valid_num = self.data_num - self.train_num
        self.x_train_names = self.xnames[:self.train_num]
        self.y_train_angles = self.yangles[:self.train_num]
        self.x_valid_names = self.xnames[-self.valid_num:]
        self.y_valid_angles = self.yangles[-self.valid_num:]
        assert self.train_num == len(self.x_train_names), "train size destn't match"
        assert self.valid_num == len(self.x_valid_names), "valid size destn't match"

    def check_and_shuffle_index(self):
        if self.shuffle:
            print("\nShuffling index ...")
            # shuffle list of angles
            l = list(zip(self.xnames, self.yangles))
            random.shuffle(l)
            self.xnames, self.yangles = zip(*l)

    def get_train_data(self):
        # X = [preprocess_image(img_to_array(load_img(i))) for i in self.x_train_names]
        X = [load_processed_image(i) for i in self.x_train_names]
        return np.asarray(X), np.asarray(self.y_train_angles)

    def get_valid_data(self):
        # X = [preprocess_image(img_to_array(load_img(i))) for i in self.x_valid_names]
        X = [load_processed_image(i) for i in self.x_valid_names]
        return np.asarray(X), np.asarray(self.y_valid_angles)

    def get_label_data(self):
        return np.asarray(self.yangles)

    def get_data_sizes(self):
        return self.train_num, self.valid_num

    def gen_data_from_dir(self, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        train_index = 0
        while 1:
            # print("gen_data_from_dir: train_index=", train_index, self.train_num)
            if train_index + self.batch_size > self.train_num:
                train_index = 0
                self.check_and_shuffle_index()
            paths = self.x_train_names[train_index: train_index + self.batch_size]
            # X = [preprocess_image(img_to_array(load_img(i))) for i in paths]
            X = [load_processed_image(i) for i in paths]
            y = [self.y_train_angles[train_index: train_index + self.batch_size]]
            train_index += self.batch_size
            yield np.array(X), np.rollaxis(np.array(y), axis=1)
