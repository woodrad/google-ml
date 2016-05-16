#! /usr/bin/env python3
#
# Copyright (C) 2016 Mathew Woodyard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

x = iris.data # features
y = iris.target # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

dtree_classifier = tree.DecisionTreeClassifier()

dtree_classifier.fit(x_train, y_train)
dtree_predictions = dtree_classifier.predict(x_test)
print(dtree_predictions)

print(accuracy_score(y_test, dtree_predictions))

kneighbors_classifier = KNeighborsClassifier()

kneighbors_classifier.fit(x_train, y_train)
kneighbors_predictions = kneighbors_classifier.predict(x_test)
print(kneighbors_predictions)

print(accuracy_score(y_test, dtree_predictions))

