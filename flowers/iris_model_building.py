import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

import pickle
pickle.dump(clf, open('iris_clf.pkl', 'wb'))