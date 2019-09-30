'''logistic implementation for binary classification
optimize using gradient descent for log loss
use sigmoid activation function
update loss using log loss
'''

import numpy as np

class LogisticGD(object):
    def __init__(self, iter, eta, random_state):
        self.n_iter = iter
        self.eta = eta
        self.random_state = random_state

    def add_dim(self, X):
        p = np.ones((X.shape[0],1))
        return np.append(X, p, axis=1)

    def net_input(self, X):
        return np.dot(X, self.w_)

    def activation_function(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -300, 300))) #clip limits the value of X.T.w to -300 to 300 to avoid overflow
    
    def fit(self, X, y):
        seed = np.random.RandomState(self.random_state)
        # init weight drawn from random gaussian with mean 0 and variance 0.01
        X = self.add_dim(X)        
        self.w_ = seed.normal(loc=0.0, scale=0.01, size=X.shape[1])
        for epoch in range(self.n_iter):
            z = self.net_input(X)
            z_out = self.activation_function(z)
            err = z_out - y 
            # update weight
            self.w_ -= np.dot(X.T, self.w_)
            # -(yi(log(h(xi))) + (1-yi)(log(1-h(xi)))
            log_loss = -np.dot(y.T, np.log(z_out)) - np.dot(1-y.T, np.log(1-z_out)) 
            self.loss_.append(log_loss)
            print('epoch:{}\terr:{}'.format(epoch, log_loss))
        return self

    def predict(self, X):
        return np.where(self.activation_function(self.net_input(self.add_dim(X))) >= 0.0, 1, -1)