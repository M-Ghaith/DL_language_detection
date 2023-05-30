# DL_language_detection
Deep learning model to classify audio clips of German, English, Spanish, French, Dutch, and Portuguese using PyTorch.


This repository contains a Convolutional Neural Network - Long Short Term Memory (CNN-LSTM) hybrid model for Speech Recognition, written in PyTorch.

# Project Description
The project implements a hybrid CNN-LSTM model, designed for recognizing spoken languages from short audio clips. The code first loads data, converts it into PyTorch tensors and applies Mel-Frequency Cepstral Coefficients (MFCC) transformation. These transformed data are then input to the CNN-LSTM model. The model is trained using a Cross Entropy Loss and the Adam optimizer. The training process is performed in batches and at the end of each epoch, both training and validation losses are reported. Post-training, the model is evaluated on a test dataset. The trained model is saved for further use.

Additionally, the script visualizes the model output using PCA and creates a diagram of the model architecture using TorchViz.


## Dependencies
numpy
torch
sklearn
matplotlib
torchaudio
torchviz
Please install the required dependencies using the following command:

pip install numpy torch torchvision torchaudio sklearn matplotlib torchviz

## Data

## Usage

## Model
The model architecture is a combination of Convolutional Neural Networks (CNN) and Long Short Term Memory (LSTM). The CNN part is used for feature extraction from the MFCC transformed audio clips, while the LSTM part uses these features for sequence learning. The output of the model is a softmax distribution over the six classes, representing the six languages.

Please note, currently the model doesn't handle sequences of arbitrary length due to the structure of the CNN layers. To make it handle variable-length sequences, we could eliminate the CNN layers and use only LSTM layers or incorporate an adaptive pooling layer after the CNN layers.

## License
This project is licensed under the terms of the MIT license.