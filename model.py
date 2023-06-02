import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.6)
        self.lstm = nn.LSTM(64, 128, batch_first=True, dropout=0.6)
        self.fc = nn.Linear(128, 6)  # we have 6 classes


    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Dropout layer
        x = self.dropout(x)

        # Reshape output from conv layers for input to LSTM
        x = x.view(x.size(0), x.size(3), x.size(1) * x.size(2))

        # LSTM layer
        x, _ = self.lstm(x)

        # Use output from last time step
        x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)

        return x