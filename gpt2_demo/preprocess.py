import os
import requests
from tqdm import tqdm


def download_gpt2(data_path, model_name='117M'):
	"""Downloads the GPT-2 model into the current directory
	from Google Cloud Storage.

	Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
	"""

	untrained_filenames = ['checkpoint', 'encoder.json', 'hparams.json',
					 'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
					 'model.ckpt.meta', 'vocab.bpe']

	trained_filenames = ['checkpoint', 'encoder.json', 'hparams.json',
					 'model-1001.data-00000-of-00001', 'model.1001.index',
					 'model.1001.meta', 'vocab.bpe']

	for filename in trained_filenames:

		untrained_model = "https://storage.googleapis.com/gpt-2/models/" + model_name + "/"
		trained_model = "https://storage.googleapis.com/myelin-gpt-2/models/shakespeare/" + model_name + "/"

		r = requests.get(trained_model + filename, stream=True)

		with open(os.path.join(data_path, filename), 'wb') as f:
			file_size = int(r.headers["content-length"])
			chunk_size = 1000
			with tqdm(ncols=100, desc="Fetching " + filename,
					  total=file_size, unit_scale=True) as pbar:
				for chunk in r.iter_content(chunk_size=chunk_size):
					f.write(chunk)
					pbar.update(chunk_size)


def download_text(data_path):
	url = "http://www.gutenberg.org/files/100/100-0.txt"
	r = requests.get(url, stream=True)

	filename = "shakespeare.txt"

	with open(os.path.join(data_path, filename), 'wb') as f:
		file_size = int(r.headers["content-length"])
		chunk_size = 1000
		with tqdm(ncols=100, desc="Fetching " + filename,
				  total=file_size, unit_scale=True) as pbar:
			for chunk in r.iter_content(chunk_size=chunk_size):
				f.write(chunk)
				pbar.update(chunk_size)


if __name__ == '__main__':
	data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	download_gpt2(data_path)
	download_text(data_path)
