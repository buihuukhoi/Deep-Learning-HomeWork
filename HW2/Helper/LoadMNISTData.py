from tensorflow.examples.tutorials.mnist import input_data

def load_data(mode='train'):
    """
    Function to load the MNIST data from tensorflow
    :param mode: train of test
    :return: images and labels
    """
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    if mode == 'train':
        x_train, y_train = mnist.train.images, mnist.train.labels
        x_validation, y_validation = mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_validation, y_validation
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
        return x_test, y_test

    def reshape(x, y):
        """
        Function to reformat the data then it can be used by convolutional layers
        :param x: input array of images
        :param y: corresponding labels of images
        :return: reshaped input images and labels
        """
        return None
