import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetwork(nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(self.model.children())[:-1])) # remove end of model
        # TODO: Double check the below 2048 input size
        self.fc = nn.Linear(2048, 5)

        self.flatten = nn.Flatten() 

    def forward(self, x):
        # freeze weights
        with torch.no_grad():
            features = self.newmodel(x)
            print(features.size())
            features = self.flatten(features)
        prediction = self.fc(features)
        return prediction

