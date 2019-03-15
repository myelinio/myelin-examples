import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "blas.ldflags=\"-L/usr/lib/openblas-base -lopenblas\""

from myelin_model.cf_model import CFModel
from myelin_model.utils import load_obj
import numpy as np
import pandas as pd
import sys
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
data_path = os.environ.get('DATA_PATH') or '/tmp/data/'


class RNNRecommender(object):
	def __init__(self):
		model_parameters = load_obj(model_path, "model_parameters")
		self.trained_model = CFModel(model_parameters["max_userid"], model_parameters["max_movieid"],
									 model_parameters["k_factors"])
		self.trained_model.load_weights(os.path.join(model_path, 'weights.h5'))
		self.trained_model._make_predict_function()
		self.class_names = ["class:rating"]
		self.movie_lookup = self.load_movie_names()
		self.movie_ids = list(self.movie_lookup.keys())
		self.c = metric.MetricClient()

	@staticmethod
	def load_movie_names():
		movies = pd.read_csv(data_path + "movies.csv", sep='\t', index_col=0)[['movie_id', 'title']]
		return dict(zip(movies.movie_id, movies.title))

	def predict_rating(self, row):
		user_id = row[0]
		movie_id = row[1]
		return self.trained_model.rate(user_id - 1, movie_id - 1)

	def predict_all_movies(self, user_ids):
		user_vector = np.repeat(user_ids, len(self.movie_ids))
		movie_vector = np.array(self.movie_ids * len(user_ids))
		# Embedded id is currently id - 1
		pred = self.trained_model.batch_rate(user_vector - 1, movie_vector - 1)
		pred_df = pd.DataFrame({"user_id": user_vector, "movie_id": movie_vector, "rating": pred.reshape(-1)})
		top_recommendation = pred_df.groupby("user_id")[["movie_id", "rating"]].apply(
			lambda dfg: dfg.nlargest(5, 'rating'))
		return top_recommendation

	def get_movie_names(self, movie_ids):
		return [self.movie_lookup[m] for m in movie_ids]

	def predict(self, users, feature_names):
		recs = self.predict_all_movies(users)
		recommended_movies = [
			list(zip(self.get_movie_names(recs.ix[u]['movie_id'].values), recs.ix[u]['rating'].values)) for u in users]
		return recommended_movies

	def send_feedback(self, features, feature_names, reward, truth):
		print("Posting reward: %s" % reward, file=sys.stderr)
		self.c.post_update("recommender_deploy_accuracy", reward)
