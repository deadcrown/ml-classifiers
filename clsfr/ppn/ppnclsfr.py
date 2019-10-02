# an object oriented API for Perceptron classifier using only numpy
# supports fit and predict method
# uses no learning rate and initializes weight to 0
# number of iterations to find the correct decision boundaary is n_iter 

import numpy as np

class Perceptron(object):
    """Perceptron classifier:

    Parameters
    ----------
    n_iter: int
        Maximum number of epochs

    Attributes
    ----------
    w_: 1d-array
        Weights of the hyperplane after fitting
    err_: list
        number of misclassified data points in each iteration over the training dataset
    """

    def __init__(self, n_iter=100): ## constructor for default n_iter
        self.n_iter = n_iter

    def net_input(self, X):
        """Calculate net input <w,X> where w already includes the bias term"""
        return np.dot(X, self.w_[1:]) + self.w_[0] ##get value of <w,X> 

    def predict(self, X):
        """Return 1 or -1 based on the value of net_input <w,X> 
        For test point return +1 or -1 based on the value of <w,X> 
        """
        return np.where(self.net_input(X) >=0.0, 1, -1)

    def fit(self, X, y):
        """Fit training data
        Update rule is based on addition of net_input(X) in case data points lie on opposite of sign(hyperplane)
        
        Parameters
        ----------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors with number of samples is n_samples 
            and number of features in n_features
        y: {array-like}, shape=[n_samples]
            training label vector
        """
        
        self.w_ = np.zeros(1+X.shape[1])  ## weight vector that defines the hyperplane
        self.err_ = [] ##number of misclassified data points per epoch

        for _ in range(self.n_iter):
            err_pts  = 0 # if this goes to 0 then return classifier
            for xi, y_label in zip(X,y):
                wx = int(self.net_input(xi)) ## get direction of training point wrt to current hyperplane
                y_label = y_label.astype(np.float)
                # print('wx:{}\nxi:{}\ny_label:{}'.format(wx, xi, type(y_label)))
                if y_label*wx <= 0.0:
                    self.w_[1:] += y_label*xi ##update weights for all data points
                    self.w_[0] += y_label*0.01 ##adding small factor to reduce bias
                    err_pts += 1
            self.err_.append([err_pts])
            if err_pts == 0:
                return self

        return self
