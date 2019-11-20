from tensorflow.keras import datasets, layers, models
from Helper import LoadMNISTData as MNIST

# load data
x_train, y_train, x_validation, y_validation = MNIST.load_data('train')  # 55000x28x28x1, 55000x10, 5000x28x28x1, 5000x10
x_test, y_test = MNIST.load_data('test')  # 10000x28x28x1, 10000x10

# create the network layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='SAME'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='SAME'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
#model.add(layers.Dropout(0.2))  # Dropout to combat overfitting in neural networks
model.add(layers.Dense(10, activation="softmax"))
#model.build()
model.summary()

#train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_validation, y_validation))
#evaluate model
loss_result, accuracy_result = model.evaluate(x_test, y_test)
print('Testing Loss is {}'.format(loss_result))
print('Testing Accuracy is {}'.format(accuracy_result))

