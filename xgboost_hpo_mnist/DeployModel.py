import os
import pickle
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployModel(object):

	def __init__(self):
		self.model = pickle.load(open(os.path.join(model_path, 'sk.pkl'), 'rb'))
		self.c = metric.MetricClient()

	def predict(self, X, feature_names):
		predictions = self.model.predict(X)
		return predictions

	def send_feedback(self, features, feature_names, reward, truth):
		res = self.c.post_update("deploy_accuracy", reward)
		print("Posted metric with status code: %s" % res.status_code)


if __name__ == '__main__':
	d = DeployModel()
