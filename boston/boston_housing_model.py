import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=['MEDV'])

model = RandomForestRegressor()
model.fit(X, Y)

import pickle
pickle.dump(model, open('boston_housing_reg.pkl', 'wb'))
