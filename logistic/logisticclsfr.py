'''logistic implementation for binary classification
optimize using gradient descent for log loss
use sigmoid activation function
update loss using log loss
'''

import numpy

class LogisticGD(object):
    def __init__(self, iter, eta, random_state):
        self.n_iter = iter
        self.eta = eta
        self.random_state = random_state

    def add_dim(self, X):
        pass

    def net_input(self):
        pass

    def activation_function(self):
        pass
    
    def fit(self):
        pass

    def predict(self):
        pass