'''basic ensemble method to get average label for a set of classifiers
for fitting ensemble fit each base classifier and append the fitted classifier to trained clsfr list
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
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators


class MajorityVoteEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, clsfrs, wt=None, vote='ClassLabel'):
        self.clsfrs = clsfrs
        self.wt = wt
        self.vote = vote

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
        '''prediction in ensemble is mean prediction of all base classifiers
        in case the base classifier predict class labels for a test point then for ensemble take the weighted average for each classifier prediction 
        for a given test point, np.argmax(np.bincount(X, weight=base_clsfr_wts)) in case voting is done on class labels
        if the base classifiers predict calibrated probability directly(eg LR) then the average predicted label for the ensemble is weighted average of base classifier probability
        for a test point, np.argmax(np.average(base_predicttions, weight=base_clsfr_wts, axis=0)) gives the average predicted label
        '''
        # prediction for vote=ClassLabel
        if self.vote == 'ClassLabel':
            pred_l = []
            # get predictions
            for _clf in self.clsfrs:
                pred = _clf.predict(X)
                pred_l.append(pred)
            pred_arr = np.asarray(pred_l).T # use transpose to get prdictions for all classifiers row wise to do a np.bincount() row wise
            # get average prediction for each class label for each classifier
            # avg_ is a 1d array of length n_classes ie weighted bincount for each classifier prediction for each target label 
            avg_ = np.apply_along_axis(lambda x: np.bincount(x, weight=self.wt), axis=1, array=pred_arr)
            maj_vote = np.argmax(avg_, axis=1)
        else:
            # get prediction if base classifier gives probabilities
            # call predict_proba for X
            avg_ = self.predict_proba(X)
            maj_vote = np.argmax(avg_, axis=1)
        # get original class labels by inverse transform
        maj_label = self.lblenc_.inverse_transform(maj_vote)
        return maj_label

    def predict_proba(self, X):
        '''function to iteratively call predict_proba() for each base classifier
        average out the prediction for each class label by taking a weighted average on axis=0
        returns avg weighted probability for all class labels for all classifiers
        '''
        clf_l = []
        for _clf in self.clsfrs:
            pred = _clf.predict_proba(X)
            pred_l.append(pred)
        pred_arr = np.asarray(pred_l)
        avg_ = np.average(pred_arr, weight=self.wt, axis=0)
        return avg_