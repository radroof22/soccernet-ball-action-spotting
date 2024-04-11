import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 21 * 42, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Dropout(0.5),
            nn.Linear(60, 13),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x