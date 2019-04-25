import os

from six.moves.urllib.request import urlretrieve
import zipfile
import pandas as pd


def download_movielens_data(name, path):
    dataset = dict(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip')

    print('Trying to download dataset from ' + dataset["url"] + '...')
    tmp_file_path = os.path.join(path, 'tmp.zip')
    urlretrieve(dataset["url"], tmp_file_path)

    with zipfile.ZipFile(tmp_file_path, 'r') as tmp_zip:
        tmp_zip.extractall(path)

    os.remove(tmp_file_path)
    print('Done! Dataset', name, 'has been saved to',
          os.path.join(path, name))


def process_data(base_path, user_data_file, movie_data_file, rating_data_file,
                 users_csv_file, movies_csv_file, ratings_csv_file):
    movielens_dir = os.path.join(base_path, "ml-1m")
    AGES = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
    OCCUPATIONS = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                   4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                   7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                   12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                   17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

    # Read the Ratings File
    ratings = pd.read_csv(os.path.join(movielens_dir, rating_data_file),
                          sep='::',
                          engine='python',
                          encoding='latin-1',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    print(len(ratings), 'ratings loaded')

    # Set max_userid to the maximum user_id in the ratings
    max_userid = ratings['user_id'].drop_duplicates().max()
    # Set max_movieid to the maximum movie_id in the ratings
    max_movieid = ratings['movie_id'].drop_duplicates().max()

    # Process ratings dataframe for Keras Deep Learning model
    # Add user_emb_id column whose values == user_id - 1
    ratings['user_emb_id'] = ratings['user_id'] - 1
    # Add movie_emb_id column whose values == movie_id - 1
    ratings['movie_emb_id'] = ratings['movie_id'] - 1

    # Save into ratings.csv
    ratings.to_csv(os.path.join(base_path, ratings_csv_file),
                   sep='\t',
                   header=True,
                   encoding='latin-1',
                   columns=['user_id', 'movie_id', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
    print('Saved to', ratings_csv_file)

    # Read the Users File
    users = pd.read_csv(os.path.join(movielens_dir, user_data_file),
                        sep='::',
                        engine='python',
                        encoding='latin-1',
                        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
    users['age_desc'] = users['age'].apply(lambda x: AGES[x])
    users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])
    print(len(users), 'descriptions of', max_userid, 'users loaded.')

    # Save into users.csv
    users.to_csv(os.path.join(base_path, users_csv_file),
                 sep='\t',
                 header=True,
                 encoding='latin-1',
                 columns=['user_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])
    print('Saved to', users_csv_file)

    # Read the Movies File
    movies = pd.read_csv(os.path.join(movielens_dir, movie_data_file),
                         sep='::',
                         engine='python',
                         encoding='latin-1',
                         names=['movie_id', 'title', 'genres'])
    print(len(movies), 'descriptions of', max_movieid, 'movies loaded.')

    # Save into movies.csv
    movies.to_csv(os.path.join(base_path, movies_csv_file),
                  sep='\t',
                  header=True,
                  columns=['movie_id', 'title', 'genres'])
    print('Saved to', movies_csv_file)


data_path = os.environ.get('DATA_PATH') or '/tmp/data/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

download_movielens_data("ml-1m", data_path)
data = process_data(data_path, 'users.dat', 'movies.dat', 'ratings.dat',
                    'users.csv', 'movies.csv', 'ratings.csv')
