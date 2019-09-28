import numpy as np
import math
import sys

sys.path.append('..')
from toolbox import funclib
from toolbox.funclib import get_minkwski_dist

def fit(X_train, x_test, k=3, p=2):
    '''function to get k nearest neighbours from x_test in X_train
    return list of k tuples with instance and distance
    Parameters
    ----------
    X_train: {array_like}, shape=[n_samples, n_features]
        training dataset to get nearest neighbours from
    x_test: {array_like}, shape=[1, n_features]
        test instance
    k: int, default=3
        k nearest neighbours from test point
    p: int, default=2
        degree of minkowski distance to be used
    '''
    _nn = []
    for samp in X_train:
        d = get_minkwski_dist(samp[0], x_test, p=p, norm=norm)
        _nn.append((d, samp[0], samp[1]))
    _nn = sorted(_nn, key=lambda s: s[0])
    return _nn[:k]

def predict(nn_list, reg=False):
    '''function to get predictions based on nn labels
    returns a class label for classification and mean of neighbours for regression
    Parameters
    ----------
    nn_list: {array_like}
        list of tuples with nn from test point
    reg: boolean, default=False
        boolean flag for regression labels
    '''
    trgt_cls = [elem[2] for elem in nn_list]
    if reg == False:
        trgt_dict = {elem:0 for elem in set(trgt_cls)}
        for _ in trgt_cls:
            trgt_dict[_] += 1
        return max(trgt_dict)
    else:
        trgt_mean = np.array(trgt_cls).mean()
        return trgt_mean
