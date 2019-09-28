import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_region(X, y, clsfr, test_idx=None, res=0.2):    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),
            np.arange(x2_min, x2_max, res))

    Z = clsfr.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=cl,
                edgecolor='black')        
        # highlight test samples
        if test_idx:
                # plot all samples
                X_test, y_test = X[test_idx, :], y[test_idx]
                plt.scatter(X_test[:, 0], X_test[:, 1],
                        c='', edgecolor='black', alpha=1.0,
                        linewidth=1, marker='o',
                        s=100, label='test set')

def train_test_split(X, Y, split=0.8, norm=None, seed=None):
        '''function to create test train split based on split size
        seed is used to reinitialize the random state to some previous state
        return X_train, X_test as iterables with zip of instance, target
        Parameters
        ----------
        X: {array-like}, shape=[n_samples, n_features]
                Dataset instances
        Y: {array-like}, shape=[n_samples]
                Dataset labels
        split: float, default=0.8
                split ratio to be used for training and test split
        norm: string, default=None
                possible norm values: MAX, MEAN
        seed: int, default=None
                seed used to reinitialize random state in case results need to be reproduced        
        '''
        if type(X).__module__ != np.__name__:
                print('input needs to be a numpy array')
                return 
        if seed != None:
                np.random.seed(seed)
        msk = np.random.rand(len(X)) < split
        if norm != None:
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                if X.dtype in numerics:
                        for _ in range(int(X.shape[1])):
                                _dim = X[:,_]
                                if norm == r'MAX':
                                        _dim = (_dim - _dim.min())/(_dim.max() - _dim.min())
                                        X[:,_] = _dim
                                elif norm == r'MEAN':
                                        _dim = (_dim - _dim.mean())/(_dim.std())
                                        X[:,_] = _dim
                                else:
                                        print('use MAX or MEAN as possible norm values')
                                        return
                else:
                        print('ERROR.. use numeric columns for max regularization')
                        return
        X_train = X[msk]
        Y_train = Y[msk]
        X_test = X[~msk]
        Y_test = Y[~msk]
        return zip(X_train, Y_train), zip(X_test, Y_test)

def get_minkwski_dist(x1, x2, p=2):
    '''function to get minkowski distance between two instances
    p defnines the degree of distance(default p=2(eucledian))
    Parameters
    ----------
    x1, x2: {array_like}, shape=[1, n_features]
            instances for calculating distances
    p: int, default=2
            degree of minkowski distance
    '''
    if len(x1) != len(x2):
        print('instance size is unequal')
        return
    feat_len = len(x1)
    dist = 0
    for _ in range(feat_len):
        dist += math.pow(abs(x1[_] - x2[_]), p)
    return math.pow(dist, 1./p)