import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torchmetrics import Accuracy

import pytorch_lightning as pl
from lightning.pytorch import LightningModule

from dataLoading import DataHandler, titanicDataset


class titan_training(pl.LightningModule):
    def __init__(self, lr=0.0002):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.titan_network = titan_network()

        self.accuracy = Accuracy(task='binary').to(self.device)

    def forward(self, z):
        return self.titan_network(z)
    
    def ice_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):

        x, y = batch

        opt_Jack = self.optimizers()


        y_hat = self.titan_network(x)

        ice_loss = self.ice_loss(y_hat, y)

        opt_Jack.zero_grad()
        self.manual_backward(ice_loss)
        opt_Jack.step()

        acc = self.my_accuracy(y_hat, y)

        self.log('train_loss', ice_loss)
        self.log('train_acc', acc, prog_bar=True)
        

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.titan_network(x)

        loss = self.ice_loss(y_hat, y)

        acc = self.my_accuracy(y_hat, y)
        self.log_dict({"Val_loss": loss, "Acc": acc}, prog_bar=True) 

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.titan_network(x)

        loss = self.ice_loss(y_hat, y)

        acc = self.my_accuracy(y_hat, y)

        self.log_dict({"Acc": acc})

        

    def my_accuracy(self, y_hat, y):

        for i in range(len(y_hat)):
            if y_hat[i] < 0.5:
                y_hat[i] = 0
            else: 
                y_hat[i] = 1
        
        numCorrect = 0
        numLoops = len(y_hat)
        for i in range (numLoops):
            if y_hat[i] == y[i]:
                numCorrect = numCorrect + 1

        return (numCorrect / numLoops)

        

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_Jack = torch.optim.Adam(self.titan_network.parameters(), lr= lr)

        return opt_Jack


























class titan_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 100)
        self.do1 = nn.Dropout(p=0.2)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)

        self.fc2 = nn.Linear(100, 360)
        self.do2 = nn.Dropout(p=0.2)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)


        self.fc3 = nn.Linear(360, 100)
        self.do3 = nn.Dropout(p=0.2)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)


        self.fc4 = nn.Linear(100, 10)
        self.do4 = nn.Dropout(p=0.2)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.1)


        self.fc5 = nn.Linear(10, 1)


    def forward(self, x):

        x = F.gelu(self.fc1(x))
        x = self.do1(x)

        x = F.gelu(self.fc2(x))
        x = self.do2(x)

        x = F.gelu(self.fc3(x))
        x = self.do3(x)

        x = F.gelu(self.fc4(x))
        x = self.do4(x)

        x = torch.sigmoid(self.fc5(x))

        return x


