import json
import os
import cv2
import numpy as np
import requests
# import imageio
from PIL import Image


# url = 'http://localhost:8080/mnist-all-sklearn-axon-2979892518-983389825/predict'
# url = 'http://localhost:8080/predict'
url = 'http://localhost:5000/predict'

image = np.array(Image.open("/Users/ryadhkhsib/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/demo/f1.jpg").convert('RGB'))

jsondata = json.dumps({"data": {"ndarray": image.tolist()}})
payload = {'json': jsondata}
session = requests.session()
response = session.post(url, data=payload, headers={'User-Agent': 'test'})

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

# imageio.imwrite('outfile.jpg', image.reshape([28, 28]))
res = resize_im(image, scale=900, max_scale=1500)

print(payload)
session = requests.session()
response = session.post(url, data=payload, headers={'User-Agent': 'test'})
print(response.status_code)
print(response.text)
json_data = json.loads(response.text)
prediction = json_data["data"]["ndarray"]

print("predicted class: %s" % np.argmax(np.array(prediction)))
