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
    def __init__(self, iter=200, eta=0.01):
        self.n_iter = iter
        self.eta = eta
        pass

    def net_input(self, X, w):
        '''calculate inner product of 
        training matrix with weight vector(w.T.X)'''
        pass

    def activation_function(self, X):
        '''use identity function as activation
        return the passed vector'''
        pass
        
    def train(self, X, y):
        '''while epoch < n_iter 
        calculate errors based on activation function and train labels
        update weights
        calculate batch loss 
        update loss_epoch[]
        if loss_epoch[epoch] == 0 | n_iter
        break'''
        pass
    
    def predict(self):
        '''predict for given point based on sign(w.T.X)'''
        pass