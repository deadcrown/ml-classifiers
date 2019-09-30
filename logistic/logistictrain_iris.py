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

logistic1 = LogisticGD(eta=0.1, random_state=7, iter=300)
logistic1.fit(X_train_sc, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=logistic1, test_idx=range(80,100))
plt.legend(loc='upper left')
plt.xlabel('Standard scaled petal length(cm)')
plt.ylabel('Standard scaled petal width(cm)')
plt.title('setosa->0  versicolor->1\neta=0.1  epoch=300')
plt.savefig('logistic_eta0.1_ep300_setosa_versicolor.png')
plt.close()

logistic2 = LogisticGD(eta=0.001, random_state=7, iter=300)
logistic2.fit(X_train_sc, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=logistic2, test_idx=range(80,100))
plt.legend(loc='upper left')
plt.xlabel('Standard scaled petal length(cm)')
plt.ylabel('Standard scaled petal width(cm)')
plt.title('setosa->0  versicolor->1\neta=0.001  epoch=300')
plt.savefig('logistic_eta0.001_ep300_setosa_versicolor.png')
plt.close()

logistic3 = LogisticGD(eta=0.01, random_state=7, iter=300)
logistic3.fit(X_train_sc, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=logistic3, test_idx=range(80,100))
plt.legend(loc='upper left')
plt.xlabel('Standard scaled petal length(cm)')
plt.ylabel('Standard scaled petal width(cm)')
plt.title('setosa->0  versicolor->1\neta=0.01  epoch=300')
plt.savefig('logistic_eta0.01_ep300_setosa_versicolor.png')
plt.close()

logistic4 = LogisticGD(eta=0.0001, random_state=7, iter=300)
logistic4.fit(X_train_sc, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=logistic4, test_idx=range(80,100))
plt.legend(loc='upper left')
plt.xlabel('Standard scaled petal length(cm)')
plt.ylabel('Standard scaled petal width(cm)')
plt.title('setosa->0  versicolor->1\neta=0.0001  epoch=300')
plt.savefig('logistic_eta0.0001_ep300_setosa_versicolor.png')
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
fig.suptitle('Comparing logistic gradient descsent for different step size')
fig.subplots_adjust(hspace=0.5)
ax[0][0].plot(range(1, len(logistic1.loss_)+1), logistic1.loss_)
ax[0][0].set_xlabel('Epochs')
ax[0][0].set_ylabel('Log loss')
ax[0][0].set_title('Logistic GD - step:0.1 epoch:300')

ax[0][1].plot(range(1, len(logistic2.loss_)+1), logistic2.loss_)
ax[0][1].set_xlabel('Epochs')
ax[0][1].set_ylabel('Log loss')
ax[0][1].set_title('Logistic GD - step:0.001 epoch:300')

ax[1][0].plot(range(1, len(logistic3.loss_)+1), logistic3.loss_)
ax[1][0].set_xlabel('Epochs')
ax[1][0].set_ylabel('Log loss')
ax[1][0].set_title('Logistic GD - step:0.01 epoch:300')

ax[1][1].plot(range(1, len(logistic4.loss_)+1), logistic4.loss_)
ax[1][1].set_xlabel('Epochs')
ax[1][1].set_ylabel('Log loss')
ax[1][1].set_title('Logistic GD - step:0.0001 epoch:300')

plt.show()
plt.savefig('eta_compare_logisticGD.png')
plt.close()