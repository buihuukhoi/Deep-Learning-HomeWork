from Helper import LoadData, SupportFunctions

# =====> MNIST <=====
# load data
#x_train, y_train, x_validation, y_validation = LoadData.load_data_MNIST('train')  # 55000x28x28x1, 55000x10, 5000x28x28x1, 5000x10
#x_test, y_test = LoadData.load_data_MNIST('test')  # 10000x28x28x1, 10000x10

#file_name_without_L2 = 'saved_model.h5'
#file_name_with_L2 = 'saved_model_regularization.h5'
#images_indexes = [1, 1232]

# train model
#SupportFunctions.training_model('MNIST', x_train, y_train, x_validation, y_validation, x_test, y_test, file_name_without_L2, False)
#SupportFunctions.training_model('MNIST', x_train, y_train, x_validation, y_validation, x_test, y_test, file_name_with_L2, True)
# check my feed forward
#SupportFunctions.check_my_feed_forward(file_name_without_L2, x_test, y_test)
# show histogram of layers
#SupportFunctions.show_histogram_of_layers(file_name_without_L2)
#SupportFunctions.show_histogram_of_layers(file_name_with_L2)
# show incorrect images
#SupportFunctions.show_incorrect_correct_images('MNIST', file_name_without_L2, x_test, y_test, 'Images/MNIST/Incorrect_Images', 'Images/MNIST/Correct_Images')
# show visualized feature maps
#SupportFunctions.show_feature_maps(file_name_without_L2, x_test, images_indexes, 2, 'Images/MNIST/Visualized_Feature_Maps')


# =====> CIFAR <=====
# load data
x_train, y_train, x_validation, y_validation, x_test, y_test = LoadData.load_data_CIFAR('cifar-10-python/cifar-10-batches-py')

file_name_without_L2 = 'saved_model_CIFAR.h5'
file_name_with_L2 = 'saved_model_CIFAR_regularization.h5'
images_indexes = [1, 3]

# train model
#SupportFunctions.training_model('CIFAR', x_train, y_train, x_validation, y_validation, x_test, y_test, file_name_without_L2, False, 2)
#SupportFunctions.training_model('CIFAR', x_train, y_train, x_validation, y_validation, x_test, y_test, file_name_with_L2, True, 2)

# check my feed forward
#SupportFunctions.check_my_feed_forward(file_name_without_L2, x_test, y_test)

# show histogram of layers
#SupportFunctions.show_histogram_of_layers(file_name_without_L2)
#SupportFunctions.show_histogram_of_layers(file_name_with_L2)

# show incorrect images
#SupportFunctions.show_incorrect_correct_images(file_name_without_L2, x_test, y_test, 'Images/CIFAR/Incorrect_Images', 'Images/CIFAR/Correct_Images')

# show visualized feature maps
SupportFunctions.show_feature_maps(file_name_without_L2, x_test, images_indexes, 2, 'Images/CIFAR/Visualized_Feature_Maps')
