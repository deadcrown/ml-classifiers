'''using iris dataset for training the ensemble
using 2 features for viz purpose
comparison is made over ROC AUC as metric for each base classifier and ensemble
using three H; decision_tree, knn, logistic_regression
'''

import numpy as np
import sys

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

# prep training dataset
iris = datasets.load_iris()
X = iris.data[:, [1,2]]
y = iris.target
lblenc = LabelEncoder()
y = lblenc.fit_transform(y)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

knn_clf = KNeighborsClassifier(
    n_neighbors = 1,
    p = 2,
    metric='minkowski'
)
dt_clf = DecisionTreeClassifier(
    criterion ='entropy',
    max_depth = 1,
    random_state = 7
)
lr_clf = LogisticRegression(
    C = 0.001,
    penalty = 'l2',
    random_state = 7
)

# pipeline for knn and lr
# dt doesnt required sclaed features -- scale invariant
knn_pipe = make_pipeline(
    ['ssc', StandardScaler()],
    ['clf', knn_clf]
)
lr_pipe = make_pipeline(
    ['ssc', StandardScaler()],
    ['clf', lr_clf]
)

clsfr_name = ['LR', 'KNN', 'DT']
# get cv roc_auc over 10 folds for each base classifier
for _clf_nm, _clf in zip(clsfr_name, [lr_pipe, knn_pipe, dt_clf]):
    score = cross_val_score(
        estimator = _clf,
        X = X,
        y = y,
        cv = 10,
        scoring='roc_auc'
    )
    print('roc_auc over 10 folds for {}: {} +/- {}'.format(_clf_nm, np.mean(score), np.std(score)))

# get ensembel classifier using majority vote principle
from majorityvote_clsfr import MajorityVoteEnsemble

ens = MajorityVoteEnsemble(clsfrs=[lr_pipe, knn_pipe, dt_clf])
ens_name = 'Majority_Vote'
clsfr_name += ens_name

# get roc_auc score over 10 folds for base and ensemble classifiers
print('**********************\ncompare ensemble performance\n**********************\n')
for _clf_nm, _clf in zip(clsfr_name, [lr_pipe, knn_pipe, dt_clf, ens]):
    score = cross_val_score(
        estimator=_clf,
        X = X,
        y = y,
        cv = 10,
        scoring = 'roc_auc'
    )
    print('roc_auc over 10 folds for base {}: {} +/- {}'.format(_clf_nm, np.mean(score), np.std(score)))

