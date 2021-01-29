"""
Main entry for Experiment
"""
import os
import sys
import json
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import create_pytorch_data
from mlp import MLP
from plotting import plot_dataset, plot_series

np.random.seed(42)
pl.seed_everything(42)

global config


class Config(object):
    ATTRS = [
        'task_name',
        'experiment_name',
        'hidden_params',  # For MLP
        'learning_rate',  # For optimizer
        'momentum',
        'regularizer',
        'lambda',  # Regularizer strength
        'epochs',
        'early_stopping',
        'sigma',  # For additive Gaussian noise
    ]

    def __init__(self, **kwargs):
        for attr in Config.ATTRS:
            setattr(self, attr, kwargs.get(attr, None))

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


def create_exp_dir(
    base: Optional[str] = 'checkpoints',
    task_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    path = os.path.join(base, task_name, experiment_name)
    if not os.path.exists(path):
        os.makedirs(path)


def grid_search_network():
    """Grid search with different nh1 x nh2"""
    global config
    config = Config(**{
        'task_name': 'grid_search',
        'learning_rate': 0.1,
        'momentum': 0.9,
        'epochs': 1000,
        'early_stopping': True,
    })
    nh1 = [2, 3, 4, 5, 6]
    nh2 = [2, 3, 4, 5, 6]
    hidden_params = []
    for n1 in nh1:
        for n2 in nh2:
            hidden_params.append([(5, n1), (n1, n2)])

    grid_search_result = []

    for hp in hidden_params:
        n1, n2 = hp[0][-1], hp[-1][-1]
        setattr(config, 'experiment_name', f'h{n1}_h{n2}')
        setattr(config, 'hidden_params', hp)
        create_exp_dir(task_name=config.task_name, experiment_name=config.experiment_name)
        losses = run_experiment(config)

        grid_search_result.append({'nh1': n1, 'nh2': n2, **losses})

        summary_path = os.path.join('checkpoints', config.task_name, 'grid_search_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(grid_search_result, f)


def regularization_noise_search():
    global config
    config = Config(**{
        'task_name': 'reg_noise',
        'learning_rate': 0.1,
        'regularizer': 'L2',
        'momentum': 0.9,
        'lambda': 0.01,
        'epochs': 1000,
        'early_stopping': False,
    })
    nh1 = 4  # Best value from grid_search_network()
    nh2 = [3, 5, 6, 9]
    sigma = [0.05, 0.15]

    gs_results = []
    for n2 in nh2:
        for s in sigma:
            setattr(config, 'hidden_params', [(5, nh1), (nh1, n2)])
            setattr(config, 'sigma', s)
            setattr(config, 'experiment_name', f'h{n2}_s{s}')
            create_exp_dir(
                task_name=getattr(config, 'task_name'),
                experiment_name=getattr(config, 'experiment_name')
            )
            losses = run_experiment(config)

            gs_results.append({'nh2': n2, 'sigma': s, **losses})
            summary_path = os.path.join('checkpoints', config.task_name, 'reg_noise_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(gs_results, f)


def regularization_weights():
    global config
    config = Config(**{
        'task_name': 'reg_weights',
        'learning_rate': 0.1,
        'regularizer': 'L2',
        'momentum': 0.9,
        'epochs': 1000,
        'early_stopping': False,
        'hidden_params': [(5, 4), (4, 5)]
    })
    lambdas = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    sigmas = [0.05, 0.1, 0.15]

    for la in lambdas:
        for sigma in sigmas:
            setattr(config, 'experiment_name', f'lambda_{la}_sigma{sigma}')
            setattr(config, 'lambda', la)
            setattr(config, 'sigma', sigma)

            create_exp_dir(
                task_name=getattr(config, 'task_name'),
                experiment_name=getattr(config, 'experiment_name')
            )
            losses = run_experiment(config)


def run_experiment(config: Config):
    # Dirty, but who's care
    inspect_weights = 'weights' in getattr(config, 'task_name', '')

    # 1. Create the model
    model = MLP(hidden_params=config.hidden_params, configs=config)
    if inspect_weights:

    # 2. Load the datasets
    train_loader, val_loader, test_loader = create_pytorch_data()

    # 3. Setup checkpoint params
    ckpt_dir = os.path.join('checkpoints', config.task_name, config.experiment_name)
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor='val_loss', mode='min')

    # 4. Setup training callback
    callbacks = [checkpoint_cb]
    if config.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10))

    trainer = pl.Trainer(max_epochs=config.epochs, callbacks=callbacks)

    # 5. Run the training
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    test_loss = trainer.test(model, test_loader)

    # 6. Plot prediction on dataset
    for idx, (patterns, targets) in enumerate(val_loader):
        if idx > 0:
            raise ValueError('There should be only a single test batch')

        model.eval()
        predictions = model.predict(patterns)
        t = np.arange(1201, 1401)
        plot_series(
            t, [(targets, 'Groundtruth'), (predictions, 'Predictions')],
            title='Predictions on Test Data',
            filename=os.path.join(ckpt_dir, 'test_predictions.pdf'))

    # 7. Dump the config
    exp_data = {
        **json.loads(config.to_json()),
        'best_model_path': str(checkpoint_cb.best_model_path),
        'best_model_score': float(checkpoint_cb.best_model_score.numpy())
    }

    with open(os.path.join(ckpt_dir, 'experiment_summary.json'), 'w') as fp:
        json.dump(exp_data, fp)

    return {
        'val_loss': float(checkpoint_cb.best_model_score.numpy()),
        'test_loss': test_loss[0]['test_loss']
    }


if __name__ == '__main__':
    # grid_search_network()
    regularization_noise_search()