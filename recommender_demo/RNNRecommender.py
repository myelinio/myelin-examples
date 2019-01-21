import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "blas.ldflags=\"-L/usr/lib/openblas-base -lopenblas\""

from myelin_model.cf_model import CFModel
from myelin_model.utils import load_obj
import numpy as np
import sys
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class RNNRecommender(object):
	def __init__(self):
		model_parameters = load_obj(model_path, "model_parameters")
		self.trained_model = CFModel(model_parameters["max_userid"], model_parameters["max_movieid"],
									 model_parameters["k_factors"])
		self.trained_model.load_weights(os.path.join(model_path, 'weights.h5'))
		self.trained_model._make_predict_function()
		self.class_names = ["class:rating"]
		self.c = metric.MetricClient()

	def predict_rating(self, row):
		user_id = row[0]
		movie_id = row[1]
		return self.trained_model.rate(user_id - 1, movie_id - 1)

	def predict(self, X, feature_names):
		predictions = np.apply_along_axis(self.predict_rating, axis=1, arr=X)
		return predictions

	def send_feedback(self, features, feature_names, reward, truth):
		print("Posting reward: %s" % reward, file=sys.stderr)
		self.c.post_update("recommender_deploy_accuracy", reward)
