import sys
import numpy as np
sys.path.append('..')

from toolbox.funclib import plot_decision_region
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data[:,[2,3]], iris.target, test_size=0.3, random_state=7, stratify=iris.target)
print('train:\n{}\ntest:\n{}'.format(X_train[:2], X_test[:2]))
print('target counts:\ntest:\t{}\ntrain:\t{}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

X_comb = np.vstack([X_train, X_test])
Y_comb = np.hstack([Y_train, Y_test])

# n_neighbors -> nearest neighbors to be used; p -> degree of distance used; metric -> distance metric to be used
knn = KNeighborsClassifier(
    n_neighbors=5,
    p=2,
    metric='minkowski'
)

knn.fit(X_train, Y_train)

plot_decision_region(X_comb, Y_comb, clsfr=knn, test_idx=range(105,150))
plt.xlabel('petal length(cm)')
plt.ylabel('petal width(cm)')
plt.legend(loc='upper left')
plt.savefig('knn_5nn_iris_petal_length_width.png')