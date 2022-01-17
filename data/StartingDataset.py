import torch    
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import numpy as np

class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path = ""):
        # choose the correct dataset path
        if dataset_path == "kaggle":
            self.train = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
            self.file_path = '/kaggle/input/cassava-leaf-disease-classification/train_images/'
        else:
            self.train = pd.read_csv('cassava-leaf-disease-classification/train.csv')
            self.file_path = 'cassava-leaf-disease-classification/train_images/'
        # remove all but 2500 healthy plant images to balance out the value counts
        # get a tensor of all the healthy images indices
        healthy_indices = self.train.index[self.train["label"] == 3].values
        # randomly select healthy images to delete from dataset
        delete_healthy_indices = np.random.choice(healthy_indices, size=(len(healthy_indices-2500)), replace=False)
        # delete the healthy indices that we picked
        self.train = self.train.drop(labels=delete_healthy_indices)
        pass

    def __getitem__(self, index):
        # use pandas to open the image you want 
        image_id, image_label = self.train.iloc[index] # use iloc when indexing
        file_path = self.file_path + image_id
        image = Image.open(file_path)
        image = image.resize((448, 448))
        convert_tensor = torchvision.transforms.ToTensor() # convert from pillow to tensor
        image = convert_tensor(image)

        inputs = image
        label = image_label

        return inputs, label

    def __len__(self):
        return len(self.train) # return length of dataset
