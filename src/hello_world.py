#! /usr/bin/env python3
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

from sklearn import tree

# Classifier input
features = [[140, 1], [130, 1], [150, 0], [170, 0]] # 0 = "bumpy", 1 = "smooth"
# Classifier output
labels = [0, 0, 1, 1] # 0 = "apple", 1 = "orange"

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]])) # predicts this is an orange.