"""
Multilayer perceptron
"""
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, hidden_params=None, configs=None):
        super().__init__()
        self.configs = configs

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
                    m.bias.data.fill_(1.0)

    def forward(self, x):
        h = self.layers(x)
        return self.output(h)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.configs.learning_rate, momentum=self.configs.momentum)
        return optimizer

    def get_regularization(self, regularizer='L2', strength=0.1):
        reg = torch.Tensor([0])
        for params in self.parameters():
            if regularizer == 'L1':
                reg += torch.norm(params, 1)
            elif regularizer == 'L2':
                reg += torch.norm(params, 2)
        return reg * strength

    def training_step(self, train_batch, train_idx):
        inputs, labels = train_batch
        if self.configs.sigma:  # Add noise to the data
            inputs += np.random.normal(0, self.configs.sigma)
        predictions = self.forward(inputs)
        loss = F.mse_loss(predictions, labels)
        if self.configs.regularizer:
            loss = loss * self.get_regularization(self.configs.regularizer, getattr(self.configs, 'lambda'))
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, val_idx):
        inputs, labels = val_batch
        predictions = self.forward(inputs)
        loss = F.mse_loss(predictions, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, test_idx):
        inputs, labels = test_batch
        predictions = self.forward(inputs)
        loss = F.mse_loss(predictions, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def predict(self, inputs):
        predictions = self.forward(inputs)
        return predictions.detach().numpy()

    def get_weights(self, to_numpy=True):
        weights = torch.Tensor()
        for name, parameter in self.named_parameters():
            if 'weight' in name:
                weights = torch.cat((weights, parameter.view(-1)), 0)
        if to_numpy:
            weights = weights.detach().numpy()
        return weights


if __name__ == '__main__':
    model = MLP(hidden_params=[(5, 4), (4, 5)])
    init_weights = model.get_weights()
    assert len(init_weights) == 5 * 4 + 4 * 5 + 5