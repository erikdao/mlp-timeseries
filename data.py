"""
Generate time series data
"""
import numpy as np
import torch
import torch.utils.data as data_utils


def mackey_glass(length, beta=0.2, gamma=0.1, n=10, tau=25, noise=0):
    """Return an 1-D array following MacKey Glass series"""
    x = np.zeros(length)
    x[0] = 1.5

    for i in range(0, length - 1):
        x[i + 1] = x[i] + (beta * x[i - tau]) / (1 + x[i - tau] ** n) - gamma * x[i]

    return x


def generate_data(t_start=301, t_end=1501, lags=[20, 15, 10, 5, 0], noise=0):
    """Generate a Mackey glass dataset with (t_end - t_start) data points"""
    rows = t_end - t_start
    columns = len(lags)
    inputs = np.zeros((rows, columns))

    sequence = mackey_glass(t_end + 5, noise=noise)

    for i, time in enumerate(lags):
        inputs[:, i] = sequence[0:t_end][(t_start - time): (t_end - time)]

    output = np.array(sequence[t_start + 5: t_end + 5])
    return np.array(inputs), output.reshape(output.shape[0], 1), sequence


def split_dataset(patterns, targets, n_train=800, n_val=1000, n_test=1200):
    X_train = patterns[:n_train]
    X_val = patterns[n_train:n_val]
    X_test = patterns[n_val:n_test]

    y_train = targets[:n_train]
    y_val = targets[n_train:n_val]
    y_test = targets[n_val:n_test]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_dataset(t_start=301, t_end=1501, n_train=800, n_val=200, n_test=200):
    patterns, targets, sequence = generate_data(t_start=t_start, t_end=t_end)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(patterns, targets)

    assert X_train.shape == (n_train, 5)
    assert y_train.shape == (n_train, 1)

    assert X_val.shape == (n_val, 5)
    assert y_val.shape == (n_val, 1)

    assert X_test.shape == (n_test, 5)
    assert y_test.shape == (n_test, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_pytorch_loader(X, y, batch_size=100, shuffle=True):
    patterns = torch.from_numpy(X.astype('float32'))
    targets = torch.from_numpy(y.astype('float32'))

    dataset = data_utils.TensorDataset(patterns, targets)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_pytorch_data(batch_size=100):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset()
    train_loader = create_pytorch_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_pytorch_loader(X_val, y_val, batch_size=batch_size*2, shuffle=False)
    test_loader = create_pytorch_loader(X_test, y_test, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_pytorch_data(batch_size=100)
