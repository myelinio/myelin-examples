import os
import pandas as pd
import numpy as np

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'

if not os.path.exists(data_path):
	os.makedirs(data_path)

df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
df.to_pickle(data_path + "example.pkl")

