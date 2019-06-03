import os
from sklearn import linear_model
import pandas as pd
import pickle

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
if not os.path.exists(model_path):
	os.makedirs(model_path)

print("Loading data")
df = pd.read_pickle(data_path + "example.pkl")

X = df[["B", "C", "D", "E", "F"]].values
Y = df.A

print("Training model")
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Coefficients: \n', regr.coef_)

pickle.dump(regr, open(model_path + "lr.pkl", 'wb'))
