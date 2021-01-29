"""
Main entry for Experiment
"""
import numpy as np

from data import create_dataset
from plotting import plot_dataset

np.random.seed(1)


def run_experiment():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset()
    t = np.arange(301, 1501)
    plot_dataset(t, y_train, y_val, y_test, display=True, filename='./mackey_glass_series.png')


if __name__ == '__main__':
    run_experiment()
