import os
import keras
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
task_id = os.environ.get('TASK_ID')
axon_name = os.environ.get('AXON_NAME')


class ImageClassifier(object):

	def __init__(self):
		self.model = keras.models.load_model(os.path.join(model_path, "saved_models/keras_cifar10_trained_model.h5"))
		self.c = metric.MetricClient()

	def predict(self, X, feature_names):
		predictions = self.model.predict(X)
		return predictions

	def send_feedback(self, features, feature_names, reward, truth):
		res = self.c.post_update("image_classifer_accuracy", reward)
		print("Posted metric with status code: %s" % res.status_code)

	def tags(self):
		return {'task_id': task_id, 'axon_name': axon_name}
