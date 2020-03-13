import os

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn import metrics

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "blas.ldflags=\"-L/usr/lib/openblas-base -lopenblas\""

import pandas as pd

from recommender_demo.myelin_model.utils import save_obj

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


def load_data(base_path):
    # Reading ratings file
    ratings = pd.read_csv(os.path.join(base_path, 'ratings.csv'), sep='\t', encoding='latin-1',
                          usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
    max_userid = ratings['user_id'].drop_duplicates().max()
    max_movieid = ratings['movie_id'].drop_duplicates().max()

    # Reading ratings file
    users = pd.read_csv(os.path.join(base_path, 'users.csv'), sep='\t', encoding='latin-1',
                        usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    # Reading ratings file
    movies = pd.read_csv(os.path.join(base_path, 'movies.csv'), sep='\t', encoding='latin-1',
                         usecols=['movie_id', 'title', 'genres'])

    return ratings, users, movies, max_userid, max_movieid


ratings, users, movies, max_userid, max_movieid = load_data(data_path)

print("Max user id", max_userid)
print("Max movie_id", max_movieid)

ratings = pd.merge(ratings, users, on='user_id', how='inner')

# Create training set
shuffled_ratings = ratings.sample(frac=1., random_state=42)

# Shuffling users
user_vector = shuffled_ratings['user_emb_id'].values
print('Users:', user_vector, ', shape =', user_vector.shape)

# Shuffling movies
movie_vector = shuffled_ratings['movie_emb_id'].values
print('Movies:', movie_vector, ', shape =', movie_vector.shape)

# Shuffling ratings
rating_vector = shuffled_ratings['rating'].values
print('Ratings:', rating_vector, ', shape =', rating_vector.shape)

enc = OrdinalEncoder()
ratings[['age_desc', 'occ_desc']] = enc.fit_transform(ratings[['age_desc', 'occ_desc']])

features = ratings[['age_desc', 'occ_desc', 'movie_id']].values
labels = ratings[['rating']].values

model = RandomForestRegressor()
model.fit(features, labels)

# Show the RMSE
y_pred = model.predict(features)

val_loss = np.sqrt(metrics.mean_squared_error(labels, y_pred))
print('Minimum RMSE {:f}'.format(val_loss))

save_obj(enc, model_path, 'feature_encoder')
save_obj(model, model_path, 'rf_recommender')
