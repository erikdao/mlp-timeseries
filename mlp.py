"""
Multilayer perceptron
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 4, bias=True),
            nn.Sigmoid(),
            nn.Linear(4, 4, bias=True),
            nn.Sigmoid()
        )
        self.output = nn.Linear(4, 1, bias=True)

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

