import numpy as np
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from Helper import Prediction


def training_model(image_type, x_train, y_train, x_validation, y_validation, x_test, y_test, file_name, regularization=False, epochs=50):
    # Define param
    batch_size = 256
    kernel_size = (3, 3)
    alpha = 0.0001
    input_shape = None
    # initiate input_shape
    if image_type == 'MNIST':
        input_shape = (28, 28, 1)
    elif image_type == 'CIFAR':
        input_shape = (32, 32, 3)
    # create the network layers
    model = models.Sequential()
    if not regularization:
        model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=input_shape, padding='SAME'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, kernel_size, activation='relu', padding='SAME'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation="relu"))
        #model.add(layers.Dropout(0.2))  # Dropout to combat overfitting in neural networks
        model.add(layers.Dense(10, activation="softmax"))
        model.summary()
    else:
        model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=input_shape, padding='SAME',
                  kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, kernel_size, activation='relu', padding='SAME',
                  kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation="relu", kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))
        # model.add(layers.Dropout(0.2))  # Dropout to combat overfitting in neural networks
        model.add(layers.Dense(10, activation="softmax", kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))
        model.summary()

    # train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_check_point = callbacks.ModelCheckpoint(file_name, monitor='val_categorical_accuracy', mode='max',
                                                  verbose=1)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_validation, y_validation),
                        callbacks=[model_check_point])

    # evaluate model
    loss_result, accuracy_result = model.evaluate(x_test, y_test)
    print('Testing Loss with L2 regularization = {} is {}'.format(regularization, loss_result))
    print('Testing Accuracy with L2 regularization = {} is {}'.format(regularization, accuracy_result))

    # plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training Accuracy with L2 regularization = {}'.format(regularization))
    plt.ylabel('Accuracy rate')
    plt.xlabel('Iteration')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.show()

    # Plot learning curve
    plt.plot(history.history['loss'])
    plt.title('Learning curve with L2 regularization = {}'.format(regularization))
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Cross entropy'], loc='upper right')
    plt.show()

    # get weights
    weights = model.get_weights()
    return model, weights


def check_my_feed_forward(file_name, x_test, y_test):
    model = load_model(file_name)
    weights = model.get_weights()
    # predict output
    print('==========> Prediction <==========')
    tmp_x_test = np.array([x_test[1]])
    predicted_output = model.predict(tmp_x_test)
    print('model predict: ', predicted_output)

    my_predicted_output = Prediction.predict(tmp_x_test, model.layers, weights)
    print('my predict: ', my_predicted_output)

    print('==========> One hot vector <==========')
    print('label of output: ', np.array([y_test[1]]))
    print('model predict: ', np.round(predicted_output))
    print('my predict: ', np.round(my_predicted_output))


def histogram_of_layer(layer_name, weight):
    weight = np.reshape(weight, (-1, 1))
    plt.hist(weight, bins=100)
    plt.title('Histogram of {}'.format(layer_name))
    plt.ylabel('Number')
    plt.xlabel('Value')
    plt.show()


def show_histogram_of_layers(file_name):
    model = load_model(file_name)
    weights = model.get_weights()
    layers = model.layers

    index_of_weight = 0
    for index in range(len(layers)):
        layer = layers[index]
        if 'conv2d' in layer.name or 'dense' in layer.name:
            weight = weights[index]
            histogram_of_layer(layer.name, weight)
            index_of_weight += 2
        else:
            index_of_weight += 1


def show_incorrect_correct_images(image_type, file_name_of_model, x_test, y_test, path_name_of_incorrect, path_name_of_correct):
    model = load_model(file_name_of_model)
    predict_classes = model.predict_classes(x_test)
    y_test_classes = np.argmax(y_test, axis=1)
    incorrect_indexes = []
    correct_indexes = []
    for index in range(len(predict_classes)):
        if predict_classes[index] == y_test_classes[index]:
            correct_indexes.append(index)
        else:
            incorrect_indexes.append(index)

    for element in incorrect_indexes:
        fig, ax = plt.subplots()
        if image_type == 'MNIST':
            image = x_test[element][:, :, 0]
            ax.imshow(image, cmap='gray')
        elif image_type == 'CIFAR':
            image = x_test[element]
            ax.imshow(image)
        ax.get_xaxis().set_visible(False)  # hidden axis
        ax.get_yaxis().set_visible(False)
        ax.set_title("Label: {}, Predict: {}".format(y_test_classes[element], predict_classes[element]))
        plt.savefig("{}/{}.png".format(path_name_of_incorrect, element), bbox_inches="tight")
        plt.close(fig)

    for i in range(5):
        index = correct_indexes[i]
        fig, ax = plt.subplots()
        if image_type == 'MNIST':
            image = x_test[index][:, :, 0]
            ax.imshow(image, cmap='gray')
        elif image_type == 'CIFAR':
            image = x_test[index]
            ax.imshow(image)
        ax.get_xaxis().set_visible(False)  # hidden axis
        ax.get_yaxis().set_visible(False)
        ax.set_title("Label: {}, Predict: {}".format(y_test_classes[index], predict_classes[index]))
        plt.savefig("{}/{}.png".format(path_name_of_correct, index), bbox_inches="tight")
        plt.close(fig)

    return incorrect_indexes, correct_indexes


def show_feature_maps(file_name_of_model, x_test, images_indexes, num_of_conv_layers, path_name):
    model = load_model(file_name_of_model)
    layers = model.layers
    for image_index in images_indexes:
        input_image = np.array([x_test[image_index]])
        fig, ax = plt.subplots(nrows=num_of_conv_layers, ncols=3)
        index_of_conv = 0
        for index in range(len(layers)):
            layer = layers[index]
            model_layer = Model(inputs=model.inputs, outputs=layer.output)
            if 'conv2d' in layer.name in layer.name:
                output = model_layer.predict(input_image)  # n x height x width x depth
                output = output[0]
                # add visualized Images
                ax[index_of_conv, 0].imshow(output[:, :, 0]).set_cmap("gray")
                ax[index_of_conv, 1].imshow(output[:, :, 1]).set_cmap("gray")
                ax[index_of_conv, 2].imshow(output[:, :, 2]).set_cmap("gray")
                plt.savefig("{}/{}.png".format(path_name, image_index), bbox_inches="tight")

                index_of_conv += 1
        plt.close(fig)
