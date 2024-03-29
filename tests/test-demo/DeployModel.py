import os
import pickle
from myelin import metric
import myelin.admin

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
task_id = os.environ.get('TASK_ID')


class DeployModel(object):

    def __init__(self):
        self.model = pickle.load(open(model_path + "lr.pkl", 'rb'))
        self.c = metric.MetricClient()

    def predict(self, X, feature_names):
        predictions = self.model.predict(X)
        return predictions

    def send_feedback(self, features, feature_names, reward, truth):
        self.c.post_update("deploy_accuracy", reward)

    def tags(self):
        return {'task_id': task_id}


if __name__ == '__main__':
    d = DeployModel()
