#! /usr/bin/env python3
from sklearn import tree

# Classifier input
features = [[140, 1], [130, 1], [150, 0], [170, 0]] # 0 = "bumpy", 1 = "smooth"
# Classifier output
labels = [0, 0, 1, 1] # 0 = "apple", 1 = "orange"

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]])) # predicts this is an orange.