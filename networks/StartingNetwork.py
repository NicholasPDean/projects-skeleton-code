import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # define layers
        self.c1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.c3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.c4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten() 
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.bn1d1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn1d2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 5)
        self.bn1d3 = nn.BatchNorm1d(5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.c1(x))))
        x = self.pool(F.relu(self.bn2(self.c2(x))))
        x = self.flatten(x)
        x = F.relu(self.bn1d1(self.fc1(x)))
        x = F.relu(self.bn1d2(self.fc2(x)))
        x = self.bn1d3(self.fc3(x))

        # No activation function at the end
        # nn.CrossEntropyLoss takes care of it for us
        return x