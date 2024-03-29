'''data preprocessing steps including feature scaling and feature selection
using regularization based methods and random forest feature selection'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# shape=[178,14]
# wine = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
data_pth = os.path.join(os.path.dirname(os.getcwd()), 'data')
wine = pd.read_csv(os.path.join(data_pth, 'wine.data'), header=None)
wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']
print('label counts: {}'.format(np.unique(wine['Class label'], return_counts=True)))
print(wine.head())

X = wine.iloc[:, 1:].values
Y = wine.iloc[:, 0].values

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

# ----------FEATURE SELECTION----------
# regularization method
# using logistic for feature comparison
# checking feature importance describing class label 0 in the wine dataset
# lr_coef stores the weight vector for each class label using OvR for multiclass classification
wt_, c_ = [], []
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
# each iteration stores the weight vector for all 3 classes
# can be accessed through lr.coef_[]
# wt_ matrix with columns as feature vector weights and length = number of c used
for c in np.arange(-5., 6.):
    lr = LogisticRegression(penalty='l1', C=10**c)
    lr.fit(X_train_std, Y_train)
    wt_.append(lr.coef_[0])
    c_.append(10**c)
wt_ = np.array(wt_)
fig = plt.figure()
ax = plt.subplot(111)

feat_color_zip = zip(range(wt_.shape[1]), colors)
for ix, color in feat_color_zip:
    plt.plot(c_, wt_[:,ix], color=color, label=wine.columns[ix+1])
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('C')
plt.ylabel('weights')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03), ncol=1, fancybox=True)
plt.savefig('LR_feat_selection_class0.png')
plt.close()

# feat selection method 2 Random forest
# RF doesnt require scaled/normalized features and works out of box
feat_names = wine.columns[1:]
rf = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=7, n_jobs=4)
rf.fit(X_train, Y_train)
feat_imp = rf.feature_importances_
print('Feature_Name\tImportance')
for i in range(len(feat_imp)):
    print('{}\t{}'.format(wine.columns[i+1], feat_imp[i]))

#plot rf results
indices = np.argsort(feat_imp)[::-1]
plt.bar(range(X_train.shape[1]), feat_imp[indices])
plt.xticks(range(X.shape[1]), feat_names, rotation=90)
plt.tight_layout()
plt.savefig('rf_feat_selection_class0.png')
plt.close()