'''learning and validation curves for LR on UCI wine dataset
learning curves are plotted against increasing number of train samples and then the training vs validation accuracy can be plotted
validation curves are training vs validation accuracy on the entire dataset varying against the regularization parameter lambda 
GrisSearch can be done to get the best reg parameter based on validation accuracy observed through a brute force mechanism
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data_pth = os.path.join(os.path.dirname(os.getcwd()), 'data')
wine_data = os.path.join(data_pth, 'wine.data')
wine = pd.read_csv(wine_data, header=None)

# first column is class label
X = wine.values[:,1:]
Y = wine.values[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=7)
print('label counts:\ntest:{}\ntrain:{}'.format(np.unique(Y_test, return_counts=True), np.unique(Y_train, return_counts=True)))

# define transforms\
std = StandardScaler()
pca = PCA(n_components=2)
# define estimator
lr = LogisticRegression(C=100, solver='liblinear', multi_class='ovr', penalty='l1', random_state=7)

# stock DT for fun
dt = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=7)

lr_pipe = make_pipeline(
    std,
    pca,
    lr
)

lr_pipe.fit(X_train, Y_train)
preds = lr_pipe.predict(X_test)
print('preds:\n{}'.format(preds))
print('test accuracy of lr pipeline: {0:0.2f}'.format(lr_pipe.score(X_test, Y_test)))

# cross validation score
cv_scr = cross_val_score(
    X = X_train,
    y = Y_train,
    estimator=lr_pipe,
    cv = 10,
    n_jobs = 2
)
print('stratified k-fold CV accuracy over 10 folds: {0:0.2f} +/- {1:0.2f}'.format(np.mean(cv_scr), np.std(cv_scr)))

# get learning curve for the pipeline with training and validation accuracy over increasing samples
# batch data sie can be controlled by the argument train_sizes of learning_curve() 
# eg np.linspace(0.1,1.0,10) for 10 relatively increasing batches by using 0.1 to 1. percentage as bin size for each
'''returns: 
data_size-data size used for the particular epoch; 
train_scr:accuracy over training data size for the epoch; 
test_scr: accuracy over CV for that data size epoch'''

epoch_size, train_scr, test_scr = learning_curve(
    estimator=lr_pipe,
    X = X_train,
    y = Y_train,
    train_sizes=np.linspace(0.1,1.0,10), 
    cv = 10,
    n_jobs=2
)
print('train size: {}'.format(len(X_train)))
print('epoch batch size:\n{}'.format(epoch_size))

# plot learning curve
mean_train = np.mean(train_scr, axis=1)
std_train = np.std(train_scr, axis=1)
mean_test = np.mean(test_scr, axis=1)
std_test = np.std(test_scr, axis=1)
plt.plot(epoch_size, mean_train, marker='x', label='train accuracy', color='green')
plt.fill_between(epoch_size, mean_train+std_train, mean_train-std_train, alpha=0.10, color='green')
plt.plot(epoch_size, mean_test, marker='o', label='cv accuracy', color='blue')
plt.fill_between(epoch_size, mean_test+std_test, mean_test-std_test, alpha=0.10, color='blue')
plt.ylim((0.7,1))
plt.ylabel('Accuracy')
plt.xlabel('Epoch Size')
plt.legend(loc='lower right')
plt.title('Learning rate: UCI wine dataset(142 train samples) with 10 cv')
plt.savefig('wine_train_cv_LR_learning.png')
plt.show()
plt.close()