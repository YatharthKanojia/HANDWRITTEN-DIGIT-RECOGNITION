This project consists of two programs for handwritten digit recognition using machine learning.
The first program uses a fully connected neural network (FCNN), while the second employs a convolutional neural network (CNN) for enhanced accuracy in recognizing handwritten digits from the MNIST dataset and custom image files.

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

