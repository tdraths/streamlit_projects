import pandas as pd
path = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'
dataset = pd.read_csv(path)

X = dataset.drop(['logS'], axis=1)
Y = dataset.iloc[:, -1]

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

import pickle
pickle.dump(model, open('solubility_model.pkl', 'wb'))
