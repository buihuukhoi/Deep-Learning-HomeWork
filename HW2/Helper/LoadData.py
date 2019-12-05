import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os, sys
from six.moves import cPickle
from sklearn.model_selection import train_test_split

def load_data_MNIST(mode='train', loss='categorical_crossentropy'):
    """
    Use to load the MNIST data from tensorflow
    :param mode: train of test
    :param loss: loss parameter when compile model
    :return: images and labels
    """
    mnist = None
    if loss == 'categorical_crossentropy':  # label is one_hot_vectors
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    elif loss == 'sparse_categorical_crossentropy':  # label is integers
        mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    if mode == 'train':
        x_train, y_train = mnist.train.images, mnist.train.labels  # 55000x784 55000x10
        x_validation, y_validation = mnist.validation.images, mnist.validation.labels  # 5000x784, 5000x10
        x_train = reformat(x_train)
        x_validation = reformat(x_validation)
        return x_train, y_train, x_validation, y_validation
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
        x_test = reformat(x_test)
        return x_test, y_test

def reformat(x, num_of_channel=1):
    """
    Use to reformat the data then it can be used by convolutional layers
    :param x: input array of images
    :return: reshaped input images
    """
    img_size = int(np.sqrt(len(x[0])))
    x = x.reshape((-1, img_size, img_size, num_of_channel)).astype(np.float32)
    return x


def one_hot_label(data):
    output = np.zeros((data.size, data.max()+1))
    output[np.arange(data.size), data] = 1
    return output


def normalize_data(data):
    min = np.min(data)
    max = np.max(data)
    normalized_data = (data-min)/(max-min)
    return normalized_data


def load_batch(file_name):
    with open(file_name, 'rb') as rfile:
        if sys.version_info < (3,):
            data = cPickle.load(rfile)
        else:
            data = cPickle.load(rfile, encoding='bytes')
            # decode utf8
            data_decoded = {}
            for k, v in data.items():
                data_decoded[k.decode('utf8')] = v
            data = data_decoded
    x = data['data']
    y = data['labels']

    return x, y


def load_data_CIFAR(path_name):
    x_train = []
    y_train = []

    #  load raw train data
    for i in range(1, 6):
        file_name = os.path.join(path_name, 'data_batch_{} '.format(i))
        x_train_batch, y_train_batch = load_batch(file_name)
        x_train.append(x_train_batch)
        y_train.append(y_train_batch)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    #  load raw test data
    file_name = os.path.join(path_name, 'test_batch')
    x_test, y_test = load_batch(file_name)

    #  normalize data
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    #  reshape data
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = np.reshape(x_test, (-1, 3, 32, 32))
    x_test = x_test.transpose(0, 2, 3, 1)
    y_train = y_train.reshape(-1)
    y_test = np.reshape(y_test, (-1))

    #  convert labels to one hot vector
    y_train = one_hot_label(y_train)
    y_test = one_hot_label(y_test)

    #  split data for training
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

    return x_train, y_train, x_val, y_val, x_test, y_test,