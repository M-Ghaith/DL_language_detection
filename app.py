import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import processing
import data_loading
import MFCCDataset
import model

device = processing.device

# Data Loading
X_train = data_loading.X_train
X_test = data_loading.X_test
y_train = data_loading.y_train
y_test = data_loading.y_test


# Apply Transformation
train_dataset = MFCCDataset.MFCC(X_train)
test_dataset = MFCCDataset.MFCC(X_test)

# Convert the transformation to tensor data type
X_train = torch.stack([train_dataset[i] for i in range(len(train_dataset))])
X_test = torch.stack([test_dataset[i] for i in range(len(test_dataset))]).to(device)

##
##

# Define the training parameters
learning_rate = 1e-3
num_epochs = 50
batch_size = 64

# Instantiate the models
model = model.CNN_LSTM().to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop
losses = {'train': [], 'validation': []}

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    
    # Shuffle the training data
    indices = torch.randperm(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Mini-batch training
    for i in range(0, X_train_shuffled.shape[0], batch_size):
        # Get the mini-batch
        
        inputs = X_train_shuffled[i:i+batch_size].to(device)
        labels = y_train_shuffled[i:i+batch_size].to(device)
        labels = labels.long()
        # Forward pass
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Record the training loss for the epoch
    with torch.no_grad():
        train_outputs = model(X_train.to(device))
        train_loss = criterion(train_outputs, y_train.to(device))
        losses['train'].append(train_loss.item())
    
    # Set the model to evaluation mode
    model.eval()
    
    # Compute the validation loss
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test.to(device))
        losses['validation'].append(val_loss.item())

    # Print the epoch and loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {losses['train'][-1]}, Validation Loss: {losses['validation'][-1]}")


# Testing the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total += y_test.size(0)
    correct += (predicted == y_test.to(device)).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))


