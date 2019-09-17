import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ppnclsfr import Perceptron

sys.path.append('..') # add parent_directory path for relative imports
from toolbox import funclib
from toolbox.funclib import plot_decision_region

iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
                  , header=None)
print('iris head:\n{}'.format(iris_df.head()))

# training on two features only for better visualization
# using sepal length and petal length to predict between iris-setosa and iris-virganica(iris_df[:100], feature number[0,2])

'''df.head()
    sepal_length   sepal_width    petal_length   petal_width     species
0       5.1             3.5            1.4         0.2         Iris-setosa
1       4.9             3.0            1.4         0.2         Iris-setosa
2       4.7             3.2            1.3         0.2         Iris-setosa
3       4.6             3.1            1.5         0.2         Iris-setosa
4       5.0             3.6            1.4         0.2         Iris-setosa
'''

#select setosa and versicolor
y = iris_df.iloc[0:100, 4].values
# one-hot encode class targets -- iris-setosa=+1 and iris-virginica=-1
y = np.where(y == ['Iris-setosa'], '1', '-1')

# scatter of features -- sepal_length and petal_length; sepal_width and petal_width 
# training array based on  length
X_len = iris_df.iloc[0:100, [0,2]].values 
# training array based on widths
X_wid = iris_df.iloc[0:100, [1,3]].values

#plot data -- x_axis(sepal/petal length) and y_axis(sepal/petal width)
#define scatter for each class label for length and width separately
# on observing scatter we can see that a linear decision boundary exists for either length or width feature combinations
plt.scatter(X_len[:50, 0], X_len[:50, 1], marker='s', color='red', label='setosa')
plt.scatter(X_len[50:, 0], X_len[50:, 1], marker='v', color='blue', label='virginica')
plt.xlabel('sepal length(cm)')
plt.ylabel('petal length(cm)')
plt.legend(loc='upper left')
plt.savefig('feature_length.png')
plt.close()

plt.scatter(X_wid[:50, 0], X_wid[:50, 1], marker='s', color='red', label='setosa')
plt.scatter(X_wid[50:, 0], X_wid[50:, 1], marker='v', color='blue', label='virginica')
plt.xlabel('sepal width(cm)')
plt.ylabel('petal width(cm)')
plt.legend(loc='upper left')
plt.savefig('feature_width.png')
plt.close()

# train a ppn instance on feature lengths
ppn = Perceptron() # defalut epochs=100
ppn.fit(X_len, y)
print('trained ppn classifier\nnumber of epochs required:\t{}'.format(len(ppn.err_)))
# plot number of misclassifications vs each epoch
# total number of epochs before return is len(ppn.err_) since each epoch misclassification number is a list of list
plt.plot(range(1, len(ppn.err_)+1), ppn.err_, marker='o')
plt.xlabel('#epochs')
plt.ylabel('Total misclassified points')
plt.savefig('updatesvsepoch_length.png')
plt.close()

plot_decision_region(X_len , y, clsfr=ppn)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.savefig('dec_boundary_length.png')
plt.close()