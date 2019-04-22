import os

from myelin import metric

import cfg
import predict
from network import East
import tensorflow as tf
import numpy as np

saved_model_weights_file_path = os.environ.get('MODEL_PATH') or '/tmp/model/'

global graph
graph = tf.get_default_graph()


class DeployEASTModel(object):

    def __init__(self):
        east = East()
        self.east_detect = east.east_network()
        self.east_detect.load_weights(saved_model_weights_file_path)
        self.c = metric.MetricClient()

    def predict(self, X, feature_names):
        X = X.astype('uint8')
        quad_im, txt_items, sub_imgs = predict.predict_np(self.east_detect, X, cfg.pixel_threshold)
        return {"img_drawed": np.array(quad_im).tolist(), "txt_items": txt_items, "sub_imgs": sub_imgs}

    def send_feedback(self, features, feature_names, reward, truth):
        res = self.c.post_update("deploy_loss", reward)
        print("Posted metric with status code: %s" % res.status_code)


if __name__ == '__main__':
    d = DeployEASTModel()

    from seldon_core.model_microservice import get_rest_microservice
    app = get_rest_microservice(d)
    app.run(host='0.0.0.0', port=5001)
