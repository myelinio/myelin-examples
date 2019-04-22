import json

import imageio
import numpy as np
import requests
from PIL import Image

url = 'http://localhost:5000/predict'

image = np.array(Image.open("/home/ryadh/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/demo/f2.jpg"))

jsondata = json.dumps({"data": {"ndarray": image.tolist()}})
payload = {'json': jsondata}
session = requests.session()
response = session.post(url, data=payload, headers={'User-Agent': 'test'})

print("Response code: %s" % response.status_code)
json_data = json.loads(response.text)
prediction = json_data["data"]["ndarray"]

imageio.imwrite('outfile.jpg', np.array(prediction['img_drawed']))