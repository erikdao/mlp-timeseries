"""
Multilayer perceptron
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, hidden_params=None):
        super().__init__()

        layers = OrderedDict()
        for idx, params in enumerate(hidden_params):
            linear_name = f'linear_{idx+1}'
            layers[linear_name] = nn.Linear(*params, bias=True)
            sigmoid_name = f'sigmoid_{idx+1}'
            layers[sigmoid_name] = nn.Sigmoid()

        self.layers = nn.Sequential(layers)

        # Get the last dim of last hidden layer
        last_dim = hidden_params[-1][-1]
        self.output = nn.Linear(last_dim, 1, bias=True)

        # Initialize weights
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1)

    def forward(self, x):
        h = self.layers(x)
        return self.output(h)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, train_idx):
        inputs, labels = train_batch
        predictions = self.forward(inputs)
        loss = F.mse_loss(predictions, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        inputs, labels = val_batch
        predictions = self.forward(inputs)
        loss = F.mse_loss(predictions, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, test_idx):
        inputs, labels = test_batch
        predictions = self.forward(inputs)
        loss = F.mse_loss(predictions, labels)
        self.log('test_loss', loss)
        return loss

    def predict(self, inputs):
        predictions = self.forward(inputs)
        return predictions.detach().numpy()

