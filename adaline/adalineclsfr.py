'''Class implementation of adaline(activation function -> Identity of z)
Recipe for implementing parametric discriminative models using batch gradient descent for optimization
0. while(n_epoch < max_epochs)
1. initialize w(hyperplane) to random gaussian or 0
2. calculate net_input(w.T.X) for the batch -> z(vector.shape(n))
3. pass this net input to activation function(adaline->Identity function; logisitc->Sigmoid); output vector.shape(n)
4. calculate error for the batch: y[] - activation_output[]
5. based on loss function used use this error to update weight(hyperplane); eg square loss w = w - X.T.err[]
6. calculate net loss; eg square loss loss[sq].sum(); append loss to loss_track[] for each epoch
7. break if net loss = 0 or max_epochs
'''

import sys
import numpy as np

class Adaline(object):
    '''Class implementation of adaline
    returns weight of the hyperplane
    Attributes
    ----------
    w_: array
        set of weights for the output classifier
    loss_: list
        loss function list over all epochs
    
    Parameters
    ----------
    iter: int
        number of epochs for fitting
    eta: float
        learning rate to be used in gradient descent
    random_state: int
        seed to initialize random key
    '''
    def __init__(self, iter=200, eta=0.01, random_state=7):
        self.n_iter = iter
        self.eta = eta
        self.random_state = random_state #to reproduce results

    def net_input(self, X):
        '''calculate inner product of 
        training matrix with weight vector(w.T.X)'''
        return np.dot(X, self.w_)

    def activation_function(self, X):
        '''use identity function as activation
        return the passed vector'''
        return X
        
    def train(self, X, y):
        '''initialize weight to random gaussian noise with mean=0 ans std_dev=0.01
        assimilate bias in weight vector by adding 1 constant dimension to X
        while epoch < n_iter 
        calculate errors based on activation function and train labels
        update weights
        calculate batch loss 
        update loss_epoch_track[]
        if loss_epoch[epoch] == 0 | n_iter
        break'''
        self.loss_ = []
        ext_dim = np.ones((X.shape(0), 1)) # column vector to assimilate bias in weight
        X = np.append(X, ext_dim, axis=1)
        rand_seed = np.random.RandomState(self.random_state)
        self.w_ = rand_seed.normal(loc=0.0, scale=0.01, size=X.shape(1)+1)
        while epoch < range(self.n_iter):
            wx = self.net_input(X, self.w_)
            z = self.activation_function(wx)
            b = z[0]*self.eta
            errors = y - z
            # update weight
            w = w - self.eta*(np.dot(X.T, errors))
            epoch_loss = (errors**2).sum()/2
            self.loss_.append(epoch_loss)
        return self
    
    def predict(self, X):
        '''predict for given point based on sign(w.T.X)
        '''
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, -1)