'''basic ensemble method to get average label for a set of classifiers
if base classifiers predict class label then average predicted label is mode of weighted average of the classifier predictions
if base classifiers predict probability then average label is the weighted probability output of each classifier for each class
'''

import sys
import os
import numpy as np
import pandas as pd

# using sklearn class BaseEstimator and CLassifierMixin for using methods get_params and set_params for hyperparameter tuning
# using labelencoder for making sure that class labels start from 0 which is important for np.argmax()
# using clone to create a deep copy of individual classifier before fitting data with the copy
# _name_estimators are used to name the classifier for identification
from sklearn.basee import clone
from sklearn.base import BaseEstimator
from sklearn.base import CLassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators


class MajorityVoteEnsemble(BaseEstimator, CLassifierMixin):

    def __init__(self, clsfrs, wt=None, vote='ClassLabel'):
        self.clsfrs = clsfrs
        self.wt = wt
        self.vote = ClassLabel

    def fit(self, X, y):
        '''fit individual classifier in self.clsfrs to the train data
        self.lblenc_ is LabelEncoder instance to ensure class names start with 0
        self.classes_ is used to store label encoding for target classes_
        append fit classifiers to self.clsfrs_'''
        self.lblenc_ = LabelEncoder()
        self.lblenc_.fit(y)
        self.classes_ = self.lblenc_.classes_
        self.clsfrs_ = []
        for _clf in self.clsfrs:
            fit_clf = clone(_clf).fit(X, self.lblenc_.transform(y))
            self.clsfrs_.append(fit_clf)
        return self

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
