import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load the MNIST dataset
data = tf.keras.datasets.mnist
from tensorflow.python.keras.metrics import accuracy

(x_train, y_train), (x_test, y_test) = data.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # Neurons with ReLU activation
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # Output layer with softmax

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Predict on new images
for x in range(1, 5):
    img = cv.imread(f'{x}.png', cv.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv.resize(img, (28, 28))  # Resize image to 28x28 pixels
    img = np.invert(img)  # Invert the image
    img = img.reshape(1, 28, 28)  # Reshape to match the input shape of the model
    img = tf.keras.utils.normalize(img, axis=1)  # Normalize the image

    prediction = model.predict(img)
    print("-------------------")
    print(f"The predicted output is: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()