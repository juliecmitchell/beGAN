import torch
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.utils import shuffle
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print (sample.values[2])
        # print (torch.from_numpy(sample.values)[2].item())
        return torch.from_numpy(sample.values)

class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=transforms.Compose([ToTensor()]), training_porcentage=0.7, shuffle_db=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data = pd.read_csv(csv_file).head(100000)
        self.file = pd.read_csv(csv_file)
        if (shuffle):
            self.file = shuffle(self.file)
        self.data = self.file.head(int(self.file.shape[0]*training_porcentage))
        self.test_data = self.file.tail(int(self.file.shape[0]*(1-training_porcentage)))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.transform:
            item = self.transform(item)
        return item

    def get_columns(self):
        return self.data.columns

class DataAtts():
    def __init__(self, file_name):
        if file_name == "original_data/peptide.csv":
            self.message = "Beta Sheet Examples"
            self.class_name = "Class"
            self.values_names = {0: "H", 1: "B"}
            self.class_len = 192
            self.fname="peptide"
        else:
            print("File not found, exiting")
            exit(1)


