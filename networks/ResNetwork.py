import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetwork(nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(self.model.children())[:-1])) # remove end of model
        self.fc = nn.Linear(512, 5)

        self.flatten = nn.Flatten() 

    def forward(self, x):
        # freeze weights
        with torch.no_grad():
            features = self.newmodel(x)
            features = self.flatten(features)
        prediction = self.fc(features)
        return prediction

