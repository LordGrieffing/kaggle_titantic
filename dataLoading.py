

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from lightning.pytorch import LightningModule


class DataHandler(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir_train, data_dir_test):
        
        super().__init__()
        self.data_dir_train = data_dir_train
        self.data_dir_test = data_dir_test
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [

            ]
        )

    def setup(self, stage=None):
        
        # Assign train/val datasets
        if stage == 'fit' or stage is None:
            titanic_full = titanicDataset(data_dir= self.data_dir_train, transform= self.transform)
            self.titanic_train, self.titanic_val = random_split(titanic_full, [800, 91])

        if stage == 'test' or stage is None:
            self.titanic_test = titanicDataset(data_dir= self.data_dir_test, transform= self.transform)

    def train_dataloader(self):
        return DataLoader(self.titanic_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.titanic_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.titanic_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)




class titanicDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = np.loadtxt(self.data_dir, delimiter=',', dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(self.data[:, 2:])
        self.y = torch.from_numpy(self.data[:, [1]])
        self.data_length = len(self.data)

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        label = self.y[idx]
        sample = self.x[idx]

        return sample, label