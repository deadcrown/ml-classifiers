'''code to check performance of different ML algorithms on the same data set 
inner tuning is done thorugh param_grid in grid search which can control tuning of the individual algorithms
outer cv_score validation can give a comparison between the best h(selcted using gridsearch) from different hypothesis space
'''
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

data_pth = os.path.join(os.path.dirname(os.getcwd()), 'data')
wine_data = os.path.join(data_pth, 'wine.data')
wine = pd.read_csv(wine_data, header=None)
# get training array
X = wine.values[:, 1:]
Y = wine.values[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=7)
print('label counts:\ntrain: {}\ntest: {}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))


# define pipeline and parameters for grid search for each learning algorithm
# H used: SVM , LR, DT
# define transformations and estimators for pipeline
pca = PCA(n_components=2)
std_sclr = StandardScaler()
lr = LogisticRegression(random_state=7, multi_class='ovr')
svm = SVC(random_state=7)
dt = DecisionTreeClassifier(random_state=7)

lr_pipe = make_pipeline(
    std_sclr,
    pca,
    lr
)

svm_pipe = make_pipeline(
    std_sclr,
    pca,
    svm
)

params_range = 10**np.arange(-7., 7.)

svm_params_grid = [
    {'svc__C': params_range,
    'svc__kernel': ['linear']},
    {'svc__C': params_range,
    'svc__kernel': ['rbf'],
    'svc__gamma': params_range}
]

lr_params_grid = [
    {'logisticregression__C': params_range,
    'logisticregression__penalty': ['l1'],
    'logisticregression__solver': ['liblinear']},
    {'logisticregression__C': params_range,
    'logisticregression__penalty': ['l2'],
    'logisticregression__solver': ['lbfgs']}
]

gs_svm = GridSearchCV(
    estimator=svm_pipe,
    param_grid= svm_params_grid,
    scoring='accuracy',
    cv=4,
    n_jobs = 2,
)

gs_lr = GridSearchCV(
    estimator=lr_pipe,
    param_grid= lr_params_grid,
    scoring='accuracy',
    cv = 4,
    n_jobs = 2
)

gs_dt = GridSearchCV(
    estimator=dt,
    param_grid= [{'max_depth': [1,2,3,4,5,6,7,8,None], 'criterion': ['gini', 'entropy']}],
    scoring='accuracy',
    cv=4,
    n_jobs = 2
)

gs_svm = gs_svm.fit(X_train, Y_train)
print('SVM grid search:\nbest svm scr: {}\nbest svm parameters: {}\n'.format(gs_svm.best_score_, gs_svm.best_params_))

gs_lr = gs_lr.fit(X_train, Y_train)
print('LR grid search:\nbest lr scr: {}\nbest lr parameters: {}\n'.format(gs_lr.best_score_, gs_lr.best_params_))

gs_dt = gs_dt.fit(X_train, Y_train)
print('DT grid search:\nbest dt scr: {}\nbest dt parameters: {}\n'.format(gs_dt.best_score_, gs_dt.best_params_))

# nested cross validation 
# comparing hypothesis space
# outer cv folds is 6, inner fold in grid search is 4
# hence a 6X4 nested cross validation
lr_cv_scr = cross_val_score(
    estimator=gs_lr,
    X = X_train,
    y = Y_train,
    cv = 6,
    scoring = 'accuracy',
    n_jobs = -1
)

svm_cv_scr = cross_val_score(
    estimator=gs_svm,
    X = X_train,
    y = Y_train,
    cv = 6,
    scoring='accuracy',
    n_jobs = -1
)

dt_cv_scr = cross_val_score(
    estimator=gs_dt,
    X = X_train,
    y = Y_train,
    cv = 6,
    scoring='accuracy',
    n_jobs = -1
)

mean_svm, std_svm = np.mean(svm_cv_scr), np.std(svm_cv_scr)
mean_lr, std_lr = np.mean(lr_cv_scr), np.std(lr_cv_scr)
mean_dt, std_dt = np.mean(dt_cv_scr), np.std(dt_cv_scr)

print('*******************************************\n')
print('\nmean cv score for SVM: {} +/- {}'.format(mean_svm, std_svm))
print('\nmean cv score for LR: {} +/- {}'.format(mean_lr, std_lr))
print('\nmean cv score for DT: {} +/- {}'.format(mean_dt, std_dt))
print('*******************************************\n')