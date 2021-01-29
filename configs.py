"""
Configs for all experiments
"""


class Config(object):
    ATTRS = [
        'task_name',
        'experiment_name',
        'hidden_params',  # For MLP
        'learning_rate',  # For optimizer
        'regularizer',
        'lambda',  # Regularizer strength
        'epochs',
        'early_stopping',
        'sigma',  # For additive Gaussian noise
    ]

    def __init__(self, **kwargs):
        for attr in Config.ATTRS:
            setattr(self, attr, kwargs.get(attr, None))
