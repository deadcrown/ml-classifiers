import sys
import numpy as np
sys.path.append("..")
from toolbox.funclib import plot_decision_region

from matplotlib import pyplot as plt
from pydotplus import graph_from_dot_data

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import export_graphviz

iris = datasets.load_iris()

# using 2 dimensions petal_length and petal_width for decision boundary visualization
# stratify=1 means that the test and train sample will have equal proportions of labels
X_train, X_test, Y_train, Y_test = train_test_split(iris.data[:,[2,3]], iris.target, test_size=0.3, random_state=1, stratify=iris.target)
print('Unique labels with counts in\nTrain:\t{}\nTest:\t{}'.format(np.unique(Y_train, return_counts=True), np.unique(Y_test, return_counts=True)))

X_comb = np.vstack([X_train, X_test])
Y_comb = np.hstack([Y_train, Y_test])

# check correlation for petal_length and petal_width[2,3]
plt.scatter(iris.data[:50,2], iris.data[:50,3], marker='s', label='setosa', color='red')
plt.scatter(iris.data[50:100,2], iris.data[50:100,3], marker='*', label='versicolor', color='green')
plt.scatter(iris.data[100:150,2], iris.data[100:150,3], marker='+', label='virginica', color='blue')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')
plt.legend(loc='upper left')
plt.savefig('iris_petal_lengthvswidth.png')
plt.close()

mean_scaler = StandardScaler()
# use fit to estimate s and var for X_train individual features
mean_scaler.fit(X_train)
mean_scaler.transform(X_train)
mean_scaler.transform(X_test)

print('scaled train:\n{}\nscaled test:\n{}'.format(X_train[:2], X_test[:2]))

tree = dt(
    criterion='entropy',
    max_depth=6,
    random_state=7
)
print(tree)

# using scaled features for better decision boundary viz
tree.fit(X_train, Y_train)

# check decision boundary for test points with 5 max depth of tree
# test labels 45 105 -> 150
plot_decision_region(X_comb, Y_comb, clsfr=tree, test_idx=range(105,150))
plt.xlabel('Petal length(cm)')
plt.ylabel('Petal Width(cm)')
plt.legend(loc='upper left')
plt.title('0->setosa   1->versicolor   2->virginica')
plt.savefig('dt_6dentropy_iris_petal_length_width.png')
plt.close()

# visualize tree
# grph_data is .dot data file to be visualized using pydotplus
grph_data = export_graphviz(
    tree,
    filled=True,
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    feature_names=['petal_length', 'petal_width']
)
grph = graph_from_dot_data(grph_data)
grph.write_png('dt_6dentropy_iris_petal_length_width_graph.png')