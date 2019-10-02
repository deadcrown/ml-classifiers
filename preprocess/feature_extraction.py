'''using sklearn PCA to project features with maximum variance onto PCA components
using 2 PCA components for decision boundary viz
feature extraction on UCI wine dataset
'''
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('..')
from toolbox.funclib import plot_decision_region

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

data_pth = os.path.join(os.path.dirname(os.getcwd()), 'data')
wine_csv = os.path.join(data_pth, 'wine.data')
wine = pd.read_csv(wine_csv, header=None)
print('if any null exist: {}'.format(wine.isna==1))
wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']
print(wine.columns)
X = wine.values[:,1:]
Y = wine.values[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=7)

# feature standardadisation
ssclr = StandardScaler()
X_train_std = ssclr.fit_transform(X_train)
X_test_std = ssclr.transform(X_test)

# PCA on to two dimensions using sklearn PCA transformer
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print('X_test_pca:\n{}'.format(X_test_pca))

# use logistic regression to find decision boundary on PCA axes
lr = LogisticRegression(penalty='l1', C=10**3)
lr.fit(X_train_pca, Y_train)
# combined array for test point results
X_comb = np.vstack([X_train_pca, X_test_pca])
Y_comb = np.hstack([Y_train, Y_test])

plot_decision_region(X_comb, Y_comb, clsfr=lr, test_idx=range(142,178))
plt.legend(loc='upper right')
plt.show()
plt.savefig('lr_dec_wine_2pca.png')
plt.close()
