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
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
