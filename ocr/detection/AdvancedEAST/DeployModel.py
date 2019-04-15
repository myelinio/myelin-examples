import os

from myelin import metric

import cfg
import predict
from network import East

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployModel(object):

    def __init__(self):
        east = East()
        self.east_detect = east.east_network()
        self.east_detect.load_weights(cfg.saved_model_weights_file_path)
        self.c = metric.MetricClient()

    def predict(self, X, feature_names):
        predictions = predict(self.east_detect, X, cfg.pixel_threshold)
        return predictions

    def send_feedback(self, features, feature_names, reward, truth):
        res = self.c.post_update("deploy_loss", reward)
        print("Posted metric with status code: %s" % res.status_code)


if __name__ == '__main__':
    d = DeployModel()
