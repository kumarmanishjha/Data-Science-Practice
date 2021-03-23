import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from classifier import Net

class ClassifierDataset(Dataset):
    #Custom dataset class to create dataloader
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        
    def __len__ (self):
        return len(self.Y)