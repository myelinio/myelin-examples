import os
import pickle
from myelin import metric
import numpy as np

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployModel3(object):

	def __init__(self):
		self.model = pickle.load(open(model_path + "lr.pkl", 'rb'))
		self.c = metric.MetricClient()

	def predict(self, features_dict):
		x_model1 = features_dict['1']
		x_model2 = features_dict['2']
		x_input = features_dict['INPUT']
		X = np.concatenate([[x_model1], [x_model2], x_input], axis=1)
		predictions = self.model.predict(X)
		return predictions

	def send_feedback(self, features, feature_names, reward, truth):
		res = self.c.post_update("deploy_accuracy", reward)
		print("Posted metric with status code: %s" % res.status_code)


if __name__ == '__main__':
	d = DeployModel3()
