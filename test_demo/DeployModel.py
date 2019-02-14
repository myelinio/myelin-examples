import os
import pickle

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployModel(object):

	def __init__(self):
		self.model = pickle.load(open(model_path + "lr.pkl", 'rb'))

	def predict(self, X, feature_names):
		predictions = self.model.predict(X)
		return predictions

	def send_feedback(self, features, feature_names, reward, truth):
		pass


if __name__ == '__main__':
	d = DeployModel()
	print(d.predict([[1, 2, 3], [2, 3, 4]], None))
