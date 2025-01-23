# ANN-Project-using-MNIST-Dataset

This repository contains a Python implementation of an Artificial Neural Network (ANN) trained on a subset of the Fashion MNIST (fMNIST) dataset to classify images of clothing items into one of ten categories.

Features
Dataset: The Fashion MNIST dataset is used, containing grayscale images (28x28 pixels) of 10 clothing categories such as shirts, pants, and shoes.
Model Architecture:
Input layer: Takes 784 features (flattened 28x28 image).
Two hidden layers with 128 and 64 neurons, respectively, using ReLU activation.
Output layer with 10 neurons (one for each class) using softmax for classification.
Optimizer: Stochastic Gradient Descent (SGD).
Loss Function: Cross-Entropy Loss.

How the Model Works
Dataset Preprocessing
The dataset is read from fmnist_small.csv.
Features are normalized by dividing pixel values by 255.
Data is split into training and testing sets (80:20 split).
Neural Network
The neural network architecture is defined using PyTorch's torch.nn module. The key layers are:

Linear Layers: Fully connected layers to map input features to the output classes.
ReLU Activation: Applied to the hidden layers to introduce non-linearity.
Softmax: Used internally in the output layer to calculate class probabilities during training.
Training
The network is trained for 100 epochs using:

Batch size: 64
Learning rate: 0.001
Loss is calculated using Cross-Entropy Loss and minimized using the SGD optimizer.
Visualizing Predictions
The script includes functionality to display a random image from the test dataset along with its predicted and true labels. 

Results
Training Loss: Logs are displayed after each epoch during training.
Test Accuracy: Achieves a test accuracy of approximately 74.50%.
Contribution
Feel free to submit issues or create pull requests for improvements!
