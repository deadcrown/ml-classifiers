import sys
import numpy as np
from sklearn import datasets

sys.path.append('..')
from toolbox.funclib import train_test_split

iris = datasets.load_iris()
tr, te = train_test_split(iris.data, iris.target) #80/20
tr, te = list(tr), list(te)
X_header = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
X_train = np.array([el[0] for el in tr])
X_test = np.array([el[0] for el in te])
Y_train = np.array([el[1] for el in tr])
Y_test = np.array([el[1] for el in te])

def impurity(x, n):
    imp = 0.
    _ncls, _ncnt = np.unique(x, return_counts=True)
    _clscnt = dict(zip(_ncls, _ncnt))
    _clsp = {_k:int(_v)/n for _k,_v in _clscnt}
    for _c, _p in _clsp:
        _remsum = [_v for _k,_v in _clsp if _k != _c].sum()
        imp += _p*(1-_remsum)
    return imp

def get_impurity(X, dim_type, imp_dict):
    for _di in range(X.shape[1]):
        _dim = X[:_di]
        _feat, _featcnt = np.unique(_dim, return_counts=True)
        _nel = _featcnt.sum()
        for _ in _feat:
            if dim_type == 'CAT':
                mask = [v for v in _dim if v == _]
            if dim_type == 'CON':
                mask = [v for v in _dim if v <= np.float64(_)]
            _featarr = _dim[mask]
            _featrgt = Y_train[mask]
            imp = impurity(_featarr, _featrgt)
            imp_dict[X_header[_di]].append((str(_), imp))
    return imp_dict

n_nodes = 100
i = 0
min = -9999
root = X_train
kd_tree = {}
while(i<n_nodes):
    imp_dict = dict.fromkeys([X_header])
    imp_dict = {k:[] for k in imp_dict}
    imp_dict = get_impurity(root, imp_dict)
    # get feat with minimum impurity
    for _k,_v in imp_dict:
        _featmin = sorted(_v, key=lambda r: r[1])[0]
        i = _featmin[1]
        if min < i:
            min = i
            k,f,i = _k, _featmin[0], _featmin[1]
    kd_tree[r'{}<={}'.format(k,f)] = []
    msk = root[:int('{}'.format(X_header.index(k))] <= np.float64(r'{}'.format(f))
    left_sub = root[msk]
    right_sub = root[~msk]


def gini_impurity(X,Y):
    '''calculates the gini impurity of the passed array 
    Parameters
    ----------
    X: {array-like}, shape=[{n_samples}:n_features]
        set to calculate impurity
    Y: {array-like}, shape=[{n_samples}]
        labels of the passed set
    '''
    imp = 0.
    _cls, _cnt = np.unique(Y, return_counts=True)
    _clscnt = dict(zip(_cls, _cnt))

    for _ in range(X.shape[0]):
        _inst = X[_]
        _target = Y[_]
        for _dim in _inst:
            msk = np.where()




def fit(X_train, imp='GINI'):

