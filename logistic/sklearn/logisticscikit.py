'''implementation of logistic using scikit
example train on iris datast with inbuilt OvR for multiclass
include edcision boundary results and 
comparisons for different hyperparamters GD loss evaluation
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from toolbox.funclib import plot_decision_region

iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data[:,[2,3]], iris.target, test_size=0.2, stratify=iris.target, random_state=7)
print('target counts:\ntrain: {}\ntest: {}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

# mean scaled
sclr = StandardScaler()
sclr.fit(X_train)
train_scl = sclr.transform(X_train)
test_scl = sclr.transform(X_test)

# combined for test point viz in decision boundary
X_comb = np.vstack([train_scl, test_scl])
Y_comb = np.hstack([Y_train, Y_test])

# C is inversely proportional to regularization parameter
# in scikit logistic you dont set step size but the regularization inverse
lr = LogisticRegression(C=100, random_state=7)
lr.fit(train_scl, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=lr, test_idx=range(130,150))
plt.show()