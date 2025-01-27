# Deep Learning Models for Fashion MNIST Classification  
### (ANN and CNN Implementations)

This repository showcases two deep learning approaches—an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN)—to classify images from the Fashion MNIST (fMNIST) dataset into ten clothing categories.

---

## Features
### Dataset  
The **Fashion MNIST** dataset, containing grayscale images (28x28 pixels) of 10 clothing categories such as shirts, pants, and shoes, is used.  

### ANN Model Architecture
- **Input Layer**: Accepts 784 features (flattened 28x28 images).
- **Hidden Layers**:  
  - Two layers with 128 and 64 neurons, respectively, using ReLU activation.  
- **Output Layer**: 10 neurons (one for each class) with softmax activation for classification.
- **Optimizer**: Stochastic Gradient Descent (SGD).  
- **Loss Function**: Cross-Entropy Loss.

### CNN Model Architecture  
Implemented using PyTorch, the CNN uses convolutional layers to efficiently capture spatial patterns in the images:  

- **Feature Extraction Layers**:  
  - Conv2D (32 filters, kernel size = 3, padding = 'same') + ReLU + BatchNorm2D  
  - MaxPool2D (kernel size = 2, stride = 2)  
  - Conv2D (64 filters, kernel size = 3, padding = 'same') + ReLU + BatchNorm2D  
  - MaxPool2D (kernel size = 2, stride = 2)  

- **Classification Layers**:  
  - Flatten + Fully Connected (64x7x7 → 128) + ReLU + Dropout(0.4)  
  - Fully Connected (128 → 64) + ReLU + Dropout(0.4)  
  - Fully Connected (64 → 10)  

---

## How the Models Work

### Dataset Preprocessing
1. **Data Source**: The dataset is loaded from `fmnist_small.csv`.  
2. **Normalization**: Pixel values are scaled between 0 and 1 by dividing by 255.  
3. **Data Splitting**: 80% training and 20% testing split.

### Training
Both models are trained for **100 epochs** with the following parameters:  
- **Batch size**: 64  
- **Learning rate**: 0.001  
- **Loss Function**: Cross-Entropy Loss  
- **Optimizer**: SGD for ANN, Adam for CNN  

### Prediction Visualization
Random test images are displayed along with their **true labels** and **predicted labels** for intuitive visualization of model performance.

---

## Results  

| Model | Test Accuracy | Key Features |  
|-------|---------------|--------------|  
| **ANN** | ~74.50% | Simpler architecture, uses fully connected layers |  
| **CNN** | ~89.75% | Leverages spatial patterns with convolutional layers |  

Training logs and confusion matrices are included for both models to visualize progress and evaluate performance.

---

## Implementation Highlights

### ANN Implementation  
The ANN implementation follows a basic feed-forward architecture using fully connected layers, suitable for simpler datasets.  

### CNN Implementation  
The CNN implementation leverages convolutional layers for extracting spatial features, batch normalization for stabilizing training, and dropout for reducing overfitting.  

