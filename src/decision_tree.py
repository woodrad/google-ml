#! /usr/bin/env python3
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus

# import dataset
iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# predict label for new flower
print(test_target)
print(clf.predict(test_data))

# visualize tree
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png("../out/dtree.png")

print(iris.feature_names, iris.target_names)
print(test_data[0], test_target[0])