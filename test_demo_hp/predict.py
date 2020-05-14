import os
import pandas as pd
import numpy as np
import json
import requests


# data = pd.DataFrame(np.random.randn(2, 4), columns=list('ABCD'))
data = pd.DataFrame(np.random.randn(2, 3), columns=list('ABC'))

url = "http://localhost:8080/predict"
for _ in range(100000):
    jsondata = json.dumps({"data": {"ndarray": data.values.tolist()}})
    payload = {'json': jsondata}
    session = requests.session()
    print(jsondata)
    response = session.post(url, data=payload, headers={'User-Agent': 'test'})
    # response = session.post(url, json={"data": {"ndarray": data.values.tolist()}}, headers={'User-Agent': 'test'})

    print(response.status_code)
    print(response.text)
    json_data = json.loads(response.text)
    prediction = json_data["data"]["ndarray"]

    print("predicted class: %s" % np.argmax(np.array(prediction)))
