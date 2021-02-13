import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
dataset = pd.read_csv(r'C:\Users\Dell\3D Objects\PROPERTY_RATE\Housing.csv')
X = dataset.iloc[:, [1,2,3,4,10]].values
y = dataset.iloc[:, 0].values

print(X)
print(y)

from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor=regressor.fit(X, y)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



pickle.dump(regressor,open('iri.pkl','wb'))