'''data preprocessing steps including feature scaling and feature selection
using regularization based methods and random forest feature selection'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

# shape=[178,14]
wine = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']
print('label counts: {}'.np.unique(wine['Class label'], return_counts=True))
print(wine.head())

X = wine.iloc[:, 1:].values
Y = wine.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=7)
print('label counts:\ntrain: {}\ntest: {}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

# feature scaling
# min-max normalization
mmx_sclr = MinMaxScaler()
X_train_norm = mmx_sclr.fit_transform(X_train)
X_test_norm = mmx_sclr.transform(X_test)

# mean standardization
mean_sclr = StandardScaler()
X_train_std = mean_sclr.fit_transform(X_train)
X_test_std = mean_sclr.transform(X_test)

print(X_train_norm, X_train_std)

# feature selection using regularization 
# using logistic for feature comparison
# wt_, c_ = [], []
# colors = ['blue', 'green', 'red', 'cyan',
#           'magenta', 'yellow', 'black',
#           'pink', 'lightgreen', 'lightblue',
#           'gray', 'indigo', 'orange']
# for c in np.arange(-5, 6):
#     lr = LogisticRegression(penalty='L1', C=10**c)
#     wt = lr.fit(X_train_std, Y_train)
#     wt_.append(wt)
#     c_.append(c)