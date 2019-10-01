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
lr0 = LogisticRegression(C=100, random_state=7)
lr0.fit(train_scl, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=lr0, test_idx=range(130,150))
plt.xlabel('Standard scaled petal length')
plt.ylabel('Standard scaled petal width')
plt.title('scikit LR with C=100')
plt.savefig('lr_C100_petal_width_length.png')
plt.close()

lr1 = LogisticRegression(C=10, random_state=7)
lr1.fit(train_scl, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=lr1, test_idx=range(130,150))
plt.xlabel('Standard scaled petal length')
plt.ylabel('Standard scaled petal width')
plt.title('scikit LR with C=10(high bias)')
plt.savefig('lr_C10_petal_width_length.png')
plt.close()

lr3 = LogisticRegression(C=10000, random_state=7)
lr3.fit(train_scl, Y_train)
plot_decision_region(X_comb, Y_comb, clsfr=lr3, test_idx=range(130,150))
plt.xlabel('Standard scaled petal length')
plt.ylabel('Standard scaled petal width')
plt.title('scikit LR with C=1000(high variance)')
plt.savefig('lr_C1000_petal_width_length.png')
plt.close()

# weights variation with parameter C in scikit LR
# selected features in train -> petal length and petal wodth
wt_, coeff_ = [], []
for c in np.arange(-5, 10):
    lr = LogisticRegression(C=10.**c, random_state=7)
    lr.fit(train_scl, Y_train)
    wt_.append(lr.coef_[1])
    coeff_.append(10.**c)

wt_ = np.array(wt_)
plt.plot(coeff_, wt_[:,0], label='petal length')
plt.plot(coeff_, wt_[:,1], label='petal width')
plt.xscale('log')
plt.xlabel('C(1/reg. parameter)')
plt.ylabel('Weight')
plt.legend(loc='upper left')
plt.savefig('weight_C_variation_sklr.png')
plt.show()
plt.close()