import os
from sklearn import linear_model
import pandas as pd
import pickle
import myelin.admin

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'

prep_task = myelin.admin.task(task_name="DataPrepTest")
print(prep_task)
assert prep_task.data_path == data_path
assert prep_task.model_path == model_path


if not os.path.exists(model_path):
	os.makedirs(model_path)

print("Loading data")
df = pd.read_pickle(data_path + "example.pkl")

X = df[["B", "C", "D"]].values
Y = df.A

print("Training model")
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Coefficients: \n', regr.coef_)

pickle.dump(regr, open(model_path + "lr.pkl", 'wb'))
