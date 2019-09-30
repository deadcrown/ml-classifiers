import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from toolbox.funclib import plot_decision_region

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logisticclsfr import LogisticGD

'''use iris dataset
binary classification setosa(0) vs versicolor(+1)
use two features for decision boundary viz'''
iris = datasets.load_iris()
X = iris.data[:100, [2,3]]
Y = iris.target[:100]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
print('label counts:\ntrain: {}\ntest: {}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

# mean standard scaler
sclr = StandardScaler()
sclr.fit(X_train)
X_train_sc = sclr.transform(X_train)
X_test_sc = sclr.transform(X_test)

#combined train and test vector for decision boundary viz
X_comb = np.vstack([X_train_sc, X_test_sc])
Y_comb = np.hstack([Y_train, Y_test])

logistic = LogisticGD(eta=0.01, random_state=7, iter=200)
logistic.fit(X_train_sc, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=logistic, test_idx=range(80,100))
