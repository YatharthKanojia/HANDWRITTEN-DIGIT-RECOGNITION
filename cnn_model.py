import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Separate features and labels
x_data = data.drop(columns=['label']).values  # Drop the 'label' column for features
y_data = data['label'].values  # Keep the 'label' column for labels

# Reshape and normalize the data
x_data = x_data.reshape(-1, 28, 28, 1)  # Reshape to 28x28x1 (grayscale images)
x_data = x_data / 255.0  # Normalize the pixel values to [0, 1]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy}")
print(f"Test Loss: {loss}")

# Predict on a sample of test images
for i in range(5):
    img = x_test[i]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    print(f"Predicted Label: {np.argmax(prediction)}, True Label: {y_test[i]}")
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)
    plt.show()
