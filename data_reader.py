import numpy as np
import pickle
from skimage import color
from scipy import ndimage


class data(object):
    def __init__(self, training_file, validation_file, testing_file):
        self.train_x, self.train_y = self.load_data(training_file)
        self.val_x, self.val_y = self.load_data(validation_file)
        self.test_x, self.test_y = self.load_data(testing_file)
        # check the num of data and label
        self.check_data()

        self.normal_grayscale()

        self.img_shape = self.train_x[0].shape
        self.n_classes = np.max(self.train_y) + 1

        self.expend_training_data()

        self.n_train = self.train_x.shape[0]
        self.n_val = self.val_x.shape[0]
        self.n_test = self.test_x.shape[0]

    def check_data(self):
        assert (len(self.train_x) == len(self.train_y))
        assert (len(self.val_x) == len(self.val_y))
        assert (len(self.test_x) == len(self.test_y))

    def print_data_info(self):
        print("Number of training examples =", self.n_train)
        print("Number of validation examples =", self.n_val)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.img_shape)
        print("Number of classes =", self.n_classes)

    def load_data(self, path):
        with open(path, mode='rb') as f:
            tmp = pickle.load(f)
        return tmp['features'], tmp['labels']

    def normal_grayscale(self):
        self.train_x = (self.train_x - 128.) / 128.
        self.val_x = (self.val_x - 128.) / 128.
        self.test_x = (self.test_x - 128.) / 128.

        # convert rgb to gray scale
        self.train_x = color.rgb2gray(self.train_x)
        self.val_x = color.rgb2gray(self.val_x)
        self.test_x = color.rgb2gray(self.test_x)

    def next_batch(self, batch_size):
        """
        Return a total of `num` random samples and labels.
        """
        idx = np.arange(0, len(self.train_x))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        data_shuffle = [self.train_x[i] for i in idx]
        labels_shuffle = [self.train_y[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)


    # Augment training data
    def expend_training_data(self):
        """
        Augment training data for
        :return:
        """
        expanded_images = np.zeros([self.train_x.shape[0] * 5, self.img_shape[0], self.img_shape[1]])
        expanded_labels = np.zeros([self.train_x.shape[0] * 5])

        counter = 0
        for x, y in zip(self.train_x, self.train_y):

            # register original data
            expanded_images[counter, :, :] = x
            expanded_labels[counter] = y
            counter = counter + 1

            # get a value for the background
            # zero is the expected value, but median() is used to estimate background's value
            bg_value = np.median(x)  # this is regarded as background's value

            for i in range(4):
                # rotate the image with random degree
                angle = np.random.randint(-15, 15, 1)
                new_img = ndimage.rotate(x, angle, reshape=False, cval=bg_value)

                # shift the image with random distance
                shift = np.random.randint(-2, 2, 2)
                new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

                # register new training data
                expanded_images[counter, :, :] = new_img_
                expanded_labels[counter] = y
                counter = counter + 1

        self.train_x = expanded_images
        self.train_y = expanded_labels
