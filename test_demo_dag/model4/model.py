import os
import pickle
from myelin import metric
import numpy as np

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployModel4(object):

    def __init__(self):
        self.model = pickle.load(open(model_path + "lr.pkl", 'rb'))
        self.c = metric.MetricClient()

    def predict(self, features_dict):
        x_model1 = features_dict['DeployModel1']
        x_model2 = features_dict['DeployModel2']
        x_model3 = features_dict['DeployModel3']
        x_input = features_dict['INPUT']
        X = np.concatenate([[x_model1], [x_model2], [x_model3], x_input], axis=1)
        predictions = self.model.predict(X)
        return predictions

    def send_feedback(self, features, feature_names, reward, truth):
        res = self.c.post_update("deploy_accuracy", reward)
        print("Posted metric with status code: %s" % res.status_code)

    def model_key(self, features_dict, model_name):
        for k, _ in features_dict.items():
            if k.startswith(model_name):
                return k
        raise Exception("model %s not found in features keys %s" % (model_name,))


if __name__ == '__main__':
    d = DeployModel4()
