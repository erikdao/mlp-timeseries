"""
Utils for plotting
"""
import types
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
np.random.seed(1)


def plot_dataset(t, train, val, test, filename: Optional[str] = None, display: Optional[bool] = False):
    """Plot the whole MacKey Glass dataset"""
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    # Setup index
    i_train = n_train
    i_val = i_train + n_val
    i_test = i_val + n_test

    fig, ax = plt.subplots(figsize=(11, 5))
    plt.plot(t[:i_train], train[:, 0], label="Train Data", linewidth=2.6)
    plt.plot(t[i_train:i_val], val[:, 0], label="Val Data", linewidth=2.6)
    plt.plot(t[i_val:i_test], test[:, 0], label="Test Data", linewidth=2.6)
    plt.axvline(t[0] + i_train, color='#333333', linestyle=':')
    plt.axvline(t[0] + i_val, color='#333333', linestyle=':')
    plt.legend()
    ax.set_title("Mackey Glass Series", fontsize=16, pad=12)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    plt.setp(ax.spines.values(), color='#374151')

    if filename:
        fig.savefig(filename, bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()


def plot_series(t, series, title: Optional[str] = None, filename: Optional[str] = None, display: Optional[bool] = False):
    fig, ax = plt.subplots(figsize=(8, 5))

    data = series

    if type(series) is not list:
        if type(series) is tuple:
            data = [series]
        else:
            data = [(series, None)]

    for (d, l) in data:
        plt.plot(t, d, label=l, linewidth=2.6)
    plt.legend()
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    plt.setp(ax.spines.values(), color='#374151')

    if filename:
        fig.savefig(filename, bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()
