import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "blas.ldflags=\"-L/usr/lib/openblas-base -lopenblas\""

from myelin_model.utils import load_obj
import numpy as np
import pandas as pd
import sys
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
data_path = os.environ.get('DATA_PATH') or '/tmp/data/'


class RFRecommender(object):
    def __init__(self):
        self.feature_encoder = load_obj(model_path, "feature_encoder")
        self.trained_model = load_obj(model_path, "rf_recommender")
        self.user_data = self.load_user_data()

        self.movie_lookup = self.load_movie_names()
        self.movie_ids = list(self.movie_lookup.keys())
        self.c = metric.MetricClient()

    @staticmethod
    def load_movie_names():
        movies = pd.read_csv(data_path + "movies.csv", sep='\t', index_col=0)[['movie_id', 'title']]
        return dict(zip(movies.movie_id, movies.title))

    @staticmethod
    def load_user_data():
        return pd.read_csv(os.path.join(data_path, 'users.csv'), sep='\t', encoding='latin-1',
                           usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    def predict_all_movies(self, user_ids):
        user_vector_features = []
        for user_id in user_ids:
            user_data = self.user_data[self.user_data['user_id'] == user_id][['age_desc', 'occ_desc']]
            user_data = self.feature_encoder.transform(user_data)
            user_vector_features.append(np.tile(user_data, len(self.movie_ids)).reshape([-1, 2]))
        user_vector_features = np.concatenate(user_vector_features, axis=0)
        user_vector = np.repeat(user_ids, len(self.movie_ids))
        movie_vector = np.array(self.movie_ids * len(user_ids))
        features = np.concatenate([user_vector_features, movie_vector.reshape([-1, 1])], axis=1)

        pred = self.trained_model.predict(features)
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


if __name__ == '__main__':
    rec = RFRecommender()
    rec.predict([5411, 5439], None)
