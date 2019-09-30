'''using the iris dataset for training
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from toolbox.funclib import plot_decision_region

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from adalineclsfr import Adaline

# prep dataris-setosa
# use petal length and petal width as two features
# use two classes iris-setosa->1 and iris-versicolor-> -1
iris = datasets.load_iris()
iris_train = iris.data[:100,[2,3]]
iris_target = iris.target[:100]

iris_target = np.where(iris_target==0, 1, -1)
X_train, X_test, Y_train, Y_test = train_test_split(iris_train, iris_target, test_size=0.2, stratify=iris_target, random_state=7)
print('target counts(70/30):\ntrain:\t{}\ntest:\t{}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

# mean feature normalization
std_sclr = StandardScaler()
std_sclr.fit(X_train)
X_train_scl = std_sclr.transform(X_train)
X_test_scl = std_sclr.transform(X_test)

# combine train and test for decision boundary viz
X_comb = np.vstack([X_train_scl, X_test_scl])
Y_comb = np.hstack([Y_train, Y_test])

# train adalineclsfr
ada = Adaline(iter=100, eta=0.01)
ada.fit(X_train_scl, Y_train)

#plot decision regions
plot_decision_region(X_comb, Y_comb, clsfr=ada, test_idx=range(80,100))
plt.legend(loc='upper left')
plt.title('1->iris-setosa   -1->iris-versicolor')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.savefig('adline_0.01_100e_iris_2class.png')
plt.show()
plt.close()