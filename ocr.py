import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np


#Model
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

num_classes = 26 + 10 + 1
model = CNN_LSTM(num_classes)

print(model)
