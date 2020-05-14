import requests
import os
import tarfile
from distutils.dir_util import copy_tree

datasets = {
    # "dailymail": "0BwmD_VLjROrfM1BxdkxVaTY2bWs",
    "cnn": "0BwmD_VLjROrfTHk4NFg2SndKcjQ",
}


def download_file_from_google_drive(id, destination, file_name):
    destination_file = os.path.join(destination, file_name)
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination_file)
    tf = tarfile.open(destination_file, mode="r")
    tf.extractall(destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    dataset_path = os.path.join(data_path, 'dataset')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for k, file_meta in datasets.items():
        download_file_from_google_drive(file_meta, data_path, '%s_stories.tgz' % k)
        copy_tree(os.path.join(data_path, '%s/stories' % k), dataset_path)

"""
# Pull and install Huggingface Transformers Repo
git clone https://github.com/huggingface/transformers && cd transformers
pip install .
pip install nltk py-rouge
cd examples/summarization

#------------------------------
# Download original Summarization Datasets. The code downloads from Google drive on Linux
wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=<CONFIRMATION CODE HERE>&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -O cnn_stories.tgz

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=<CONFIRMATION CODE HERE>&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs' -O dailymail_stories.tgz

# Unzip & untar the files
tar -xvf cnn_stories.tgz && tar -xvf dailymail_stories.tgz
rm cnn_stories.tgz dailymail_stories.tgz

# Move the articles to a single location
mkdir bertabs/dataset
mkdir bertabs/summaries_out
cp -r bertabs/cnn/stories dataset
cp -r bertabs/dailymail/stories dataset

# Select a subset of articles to summarize
mkdir bertabs/dataset2
cd bertabs/dataset && find . -maxdepth 1 -type f | head -1000 | xargs cp -t ../dataset2/


"""
