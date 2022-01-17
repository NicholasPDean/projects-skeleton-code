import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetwork(nn.Module):
    """
    Basic logistic regression on 448x448x3 images.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(self.model.children())[:-1])) # remove end of model
        self.cl = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=2)
        self.fc = nn.Linear(512, 5)

        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.cl(x))))
        # freeze weights
        with torch.no_grad():
            features = self.newmodel(x)
            features = self.flatten(features)
        prediction = self.fc(features)
        return prediction
