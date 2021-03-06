import pickle
import os


def save_obj(obj, path, name):
	print(f"Saving {name} to path {path}.")
	file_name = os.path.join(path, name + '.pkl')
	with open(file_name, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path, name):
	print(f"Reading {name} from path {path}.")
	obj = os.path.join(path, name + '.pkl')
	with open(obj, 'rb') as f:
		return pickle.load(f)
