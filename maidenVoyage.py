import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from PIL import Image

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from lightning.pytorch import LightningModule

# Local imports
from dataLoading import DataHandler, titanicDataset
from titanic_network import titan_network, titan_training



# Varaibles I always want
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)
BATCH_SIZE=4

# Random Seed
random_seed = 42
torch.manual_seed(random_seed)

# Create a callback class that will do stuff during training
class CallBackReporter(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        
        epoch = trainer.current_epoch
        train_acc = trainer.callback_metrics.get("train_acc", "N/A")
        train_loss = trainer.callback_metrics.get("train_loss", "N/A")
        print(f"Epoch {epoch} - Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")

def main():
    
    #create the data module 
    dm = DataHandler(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, data_dir_train='titantic_data/cleanTrain.csv', data_dir_test= 'titantic_data/cleanTest.csv')

    Jack_and_Rose = titan_training()

    trainer = pl.Trainer(callbacks=[CallBackReporter()], max_epochs= 15, devices=AVAIL_GPUS, accelerator='gpu', log_every_n_steps=1)
    trainer.fit(Jack_and_Rose, dm)

    trainer.test(Jack_and_Rose, dm)



















if __name__ == "__main__":
    main()