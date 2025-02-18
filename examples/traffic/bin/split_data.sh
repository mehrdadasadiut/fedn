#!./.mnist-keras/bin/python
import os
from math import floor
import fire
import numpy as np


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return np.array(result)


def split(dataset='data/traffic.npz', outdir='data', n_splits=2):
    # Load and convert to dict
    package = np.load(dataset)
    data = {}
    for key, val in package.items():
        data[key] = splitset(val, n_splits)

    # Make dir if necessary
    if not os.path.exists(f'{outdir}/clients'):
        os.mkdir(f'{outdir}/clients')

    # Make splits
    for i in range(n_splits):
        subdir = f'{outdir}/clients/{str(i+1)}'
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        np.savez(f'{subdir}/traffic.npz',
                 x_train=data['x_train'][i],
                 y_train=data['y_train'][i],
                 x_test=data['x_test'][i],
                 y_test=data['y_test'][i])


if __name__ == '__main__':
    fire.Fire(split)
