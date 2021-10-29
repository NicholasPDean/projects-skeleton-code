import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()

        # define layers
        self.c1 = nn.Conv2d(3, 6, 5, padding=2)
        self.c2 = nn.Conv2d(6, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(16 * 16 * 16, 1200)
        self.fc2 = nn.Linear(1200, 320)
        self.fc3 = nn.Linear(320, 84)
        self.fc4 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # No activation function at the end
        # nn.CrossEntropyLoss takes care of it for us
        return x

