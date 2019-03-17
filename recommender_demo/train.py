import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "blas.ldflags=\"-L/usr/lib/openblas-base -lopenblas\""

import pandas as pd
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from recommender_demo.myelin_model.cf_model import CFModel
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

model_parameters = dict(max_userid=max_userid, max_movieid=max_movieid, k_factors=10)
save_obj(model_parameters, model_path, "model_parameters")

model = CFModel(model_parameters["max_userid"], model_parameters["max_movieid"], model_parameters["k_factors"])
model.compile(loss='mse', optimizer='adamax')

# Callbacks monitor the validation loss
# Save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2),
			 ModelCheckpoint(os.path.join(model_path, 'weights.h5'), save_best_only=True)]

history = model.fit([user_vector, movie_vector], rating_vector, nb_epoch=1, validation_split=.1, verbose=2,
					callbacks=callbacks)

# Show the best validation RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print('Minimum RMSE at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))