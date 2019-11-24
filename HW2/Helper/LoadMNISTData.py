import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_data(mode='train', loss='categorical_crossentropy'):
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
