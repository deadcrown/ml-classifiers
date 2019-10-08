'''using iris dataset for training the ensemble
using 2 features for viz purpose; sepal-width, petal-length
using 2 labels to directly calculate roc_auc without using OnevsAll approach for calculating roc_auc for multicalss classification
target_class_labels: iris-versicolor(0), iris-virginica(1) 
comparison is made over ROC AUC as metric for each base classifier and ensemble
using three H; decision_tree, knn, logistic_regression
'''

import numpy as np
import sys

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

# prep training dataset
iris = datasets.load_iris()
X = iris.data[50:, [1,2]]
y = iris.target[50:]
lblenc = LabelEncoder()
y = lblenc.fit_transform(y)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

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
# Pipeline is used instaed of make_pipeline to give names to each transformation which will be used for tuning individual base in ensemble
knn_pipe = Pipeline([
    ['ssc', StandardScaler()],
    ['clf', knn_clf]
])
lr_pipe = Pipeline([
    ['ssc', StandardScaler()],
    ['clf', lr_clf]
])

clsfr_name = ['LR', 'KNN', 'DT']
all_clsfrs = [lr_pipe, knn_pipe, dt_clf]
# get cv roc_auc over 10 folds for each base classifier
for _clf_nm, _clf in zip(clsfr_name, all_clsfrs):
    score = cross_val_score(
        estimator = _clf,
        X = X,
        y = y,
        cv = 10,
        scoring='roc_auc'
    )
    print('roc_auc over 10 folds for {}: {} +/- {}'.format(_clf_nm, np.mean(score), np.std(score)))

# get ensemble classifier using majority vote principle
# import tmp_majclsfr
import majorityvote_clsfr
from majorityvote_clsfr import MajorityVoteEnsemble
# from tmp_majclsfr import MajorityVoteClassifier

ens = MajorityVoteEnsemble(classifiers=[lr_pipe, knn_pipe, dt_clf])
# ens = MajorityVoteClassifier(classifiers=[lr_pipe, knn_pipe, dt_clf])
clsfr_name += ['Majority_Vote']
all_clsfrs = [lr_pipe, knn_pipe, dt_clf, ens]

# get roc_auc score over 10 folds for base and ensemble classifiers
print('**********************\ncompare ensemble performance\n**********************\n')
for _clf_nm, _clf in zip(clsfr_name, all_clsfrs):
    score = cross_val_score(
        estimator=_clf,
        X = X,
        y = y,
        cv = 10,
        scoring = 'roc_auc'
    )
    print('roc_auc over 10 folds for base {}: {} +/- {}'.format(_clf_nm, np.mean(score), np.std(score)))
