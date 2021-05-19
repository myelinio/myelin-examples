import os
import pickle
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployModel(object):

    def __init__(self):
        self.model = pickle.load(open(model_path + "lr.pkl", 'rb'))
        self.c = metric.MetricClient()

    def predict(self, X, feature_names):
        predictions = self.model.predict(X)
        return predictions

    def send_feedback(self, features, feature_names, reward, truth):
        self.c.post_update("deploy_rmse", reward)


if __name__ == '__main__':
    d = DeployModel()
