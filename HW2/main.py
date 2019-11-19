from Helper import LoadMNISTData as MNIST

# load data
train_data, train_label, validation_data, validation_label = MNIST.load_data('train')  # 55000x784, 55000x10, 5000x784, 5000x10
test_data, test_label = MNIST.load_data('test')  # 10000x784, 10000x10
