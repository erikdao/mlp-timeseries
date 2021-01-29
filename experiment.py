"""
Main entry for Experiment
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pytorch_lightning as pl

from data import create_pytorch_data
from mlp import MLP
from plotting import plot_dataset, plot_series

np.random.seed(1)


def grid_search_network():
    pass


def run_experiment():
    pl.seed_everything(1)
    train_loader, val_loader, test_loader = create_pytorch_data()

    hidden_params = [(5, 4), (4, 4)]
    model = MLP(hidden_params=hidden_params)
    print(model)

    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)

    for idx, (patterns, targets) in enumerate(val_loader):
        if idx > 0:
            raise ValueError('There should be only a single test batch')

        model.eval()
        predictions = model.predict(patterns)
        t = np.arange(1201, 1401)
        plot_series(
            t,
            [(predictions, 'Predictions'), (targets, 'Groundtruth')],
            title='Predictions on Test Data',
            display=True,
            filename='./checkpoints/test_predictions.png')


if __name__ == '__main__':
    run_experiment()
