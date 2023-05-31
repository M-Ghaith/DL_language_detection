import numpy as np
import torch

from sklearn.model_selection import train_test_split

# Splitted data Loading
X_train = np.load("DATA.npy").astype(np.float32)
X_test = np.load("DATA.npy").astype(np.float32)
y_train = np.load('DATA.npy').astype(np.float32)
y_test = np.load('DATA.npy').astype(np.float32)

## Use the code below if the your data is not split. 

# Load the data
data_X = np.load("DATA.npy").astype(np.float32)
data_y = np.load("DATA.npy").astype(np.float32)

# Defined train percentage (for example, 0.8 for 80% training data)
train_percentage = 0.8

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, train_size=train_percentage, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_train = y_train.long()
y_test = torch.from_numpy(y_test)
y_test = y_test.long()
