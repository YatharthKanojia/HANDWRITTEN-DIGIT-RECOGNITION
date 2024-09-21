This project consists of two programs for handwritten digit recognition using machine learning.
The first program uses a fully connected neural network (FCNN), while the second employs a convolutional neural network (CNN) for enhanced accuracy in recognizing handwritten digits from the MNIST dataset and custom image files.

Features
* FCNN Model: A basic neural network that takes in 28x28 pixel images and predicts the digit.
* CNN Model: A more advanced convolutional neural network for improved image recognition accuracy.
* Prediction from Custom Images: Both programs can predict digits from custom grayscale images.
* MNIST Dataset: Both models are trained and evaluated using the MNIST dataset, with additional support for real-time or custom image inputs.

FCNN Model (fcnn_model.py):
* Uses a fully connected neural network with dense layers.
* Can predict digits from custom 28x28 grayscale images.
* Trained on the MNIST dataset with a few epochs for fast results.
* The program reads images (1.png, 2.png, etc.) in grayscale, processes them, and predicts the digit.
* Output includes the predicted digit and visualization of the input image.

CNN Model (cnn_model.py):
* Utilizes convolutional and max-pooling layers for better feature extraction and accuracy.
* Also trained on the MNIST dataset with image augmentation support.
* Capable of classifying images with higher precision and handling more complex inputs.
* The program reads a CSV dataset (train.csv) and predicts digits on a test set.
* Output includes the predicted and true labels, along with the visualized digit image.

Technologies Used:
* Python: Core programming language.
* TensorFlow/Keras: For building and training the neural network models.
* NumPy: For numerical operations and image processing.
* OpenCV: For reading and processing custom images in the FCNN model.
* Matplotlib: For visualizing data and predictions.
* Pandas: For handling CSV datasets in the CNN program.

Dataset:
Program is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).
