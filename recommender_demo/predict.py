import json
import requests

address = "35.242.185.220:80"
endpoint = "ml-recommender-all3-3795085459-2539877511"

url_predict = "http://%s/%s/predict" % (address, endpoint)
data = [[10, 2], [10, 3]]

response = requests.post(url_predict, data={'json': json.dumps({"data": {"ndarray": data}})})
print(response.text)

url_feedback = "http://%s/%s/send-feedback" % (address, endpoint)
feedback = {"response": {"meta": {"routing": {"router": 0}}, "data": {"names": ["a", "b"], "ndarray": [[1.0, 2.0]]}}, "reward": 1}
response = requests.post(url_feedback, data={'json': json.dumps(feedback)})
print(response.text)
