'''model selection and evalutaion by getting a better understanding of the expected generalization error
Methods for understanding if a model has an inherent bias or variance or both issues:
0. sklearn pipelines to for streaming workflow between transformers and estimators
1. k-fold cross validation(holdout method)
2. learning curve to analyze training and validation error against number of data samples used. 
This can tell if a high variance(Training acc is high, validation is low and big gap between training and validation) 
or a high bias problem(low training accuracy and not a big gap between training and validation)
This helps in understanding the bias-variance tradeoff
3. validation curve to analyze training and validation error against regularization parameter to get an idea about optimum reg parameter(underfitting vs overfitting)
4. GridSearch: Used to store the validation error for each value of hyperparameter defined. Select the parameters with least validation error
5. nested cross-validation: a more robust technique to estimate bias than using cross-validation
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')

from toolbox.funclib import plot_decision_region

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# load the UCI wine dataset
data_pth = os.path.join(os.path.dirname(os.getcwd()), 'data')
wine_data = os.path.join(data_pth, 'wine.data')
wine = pd.read_csv(wine_data, header=None)
print(wine.head())
print(wine.isna==1)
X_ = wine.values[:,1:]
Y_ = wine.values[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2, stratify=Y_, random_state=7)
print('unique labels:\ntrain:{}\ntest:{}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

# pipelines are meta wrappers around combining set of transformations that need to be applied to train and test points before calling predict() 
# sklearn pipeline for transformers and estimators(functions which support a fit() and transform() method and a predict() for estimators)
# all transformations are applied sequentially to all points(train and test) passed through the pipeline
# the last element of a pipeline has to be an estimator
# all pipeline objects have fit(X_train, Y_train) predict(X_test) score(X_test, Y_test)  
'''pipeline elements
1. standardscaler()
2. PCA with n_components=2
3. Logistic Regression estimator
'''
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(C=100, penalty='l1', multi_class='ovr', solver='liblinear')
)
print(pipe_lr)
pipe_lr.fit(X_train, Y_train)
pipe_lr.predict(X_test)
print('test accuracy:\t{}'.format(pipe_lr.score(X_test, Y_test)))

# using stratified k-fold cross validation which is used to get a better estimate of model performance by considering a holdout method to prevent overfitting on test dataset 
# using n_splits of 10
# this implies that create 10 folds using 9 for training and 1 for validation
# called stratified k fold because proportion of labels in train is maintained inside the folds
# hence if k = 10 we get 10 different estimates for each lambda which can then be averaged out
# k fold cross validation is done to avoid overfitting on the validation data itself
# essentially k-fold is shuffling the data for each model trainingspecial case of k-fold is leave one out validation(robust but slow)
kfold = StratifiedKFold(n_splits=10, random_state=7)
kfold = kfold.split(X_train, Y_train) #return iterable of train and test array index for each k 
scores_ = []

for ix, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], Y_train[train])
    pipe_lr.predict(X_train[test])
    scores_.append(pipe_lr.score(X_train[test], Y_train[test]))
print(scores_)

# k-fold validation accuracy is the mean of all k accuracies 
# since it is now a distribution we can alse find the std dev wrt to this mean 
print('k-fold validation accuracy {0} +/- {1:0.3f}'.format(np.mean(scores_), np.std(scores_)))

# alternative implementation of stratified k-fold using sklearn 
cv_scr = cross_val_score(
    X=X_train,
    y = Y_train,
    estimator=pipe_lr,
    cv = 10,
    n_jobs=2
)
print('sklearn cv scr:\n{}'.format(cv_scr))
print('stratifed k-fold accuracy over 10 cv:\t{}+/-{}'.format(np.mean(cv_scr), np.std(cv_scr)))