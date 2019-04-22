import os

from myelin import metric
from lib.fast_rcnn.config import cfg_from_file

from text_detect import load_tf_model, ctpn, draw_boxes

saved_model_weights_file_path = os.environ.get('MODEL_PATH') or '/tmp/model/'


class DeployCTPNModel(object):

    def __init__(self):
        cfg_from_file('./ctpn/text.yml')
        sess, net = load_tf_model('./checkpoints')
        self.sess = sess
        self.net = net

        self.c = metric.MetricClient()

    def predict(self, X, feature_names):
        X = X.astype('uint8')
        scores, boxes, img, scale = ctpn(self.sess, self.net, X)
        text_recs, img_drawed = draw_boxes(img, boxes, scale)
        return {"scores": scores.tolist(),
                "boxes": boxes.tolist(),
                "img": img.tolist(),
                "img_drawed": img_drawed.tolist(),
                "text_recs": text_recs.tolist(),
                }

    def send_feedback(self, features, feature_names, reward, truth):
        res = self.c.post_update("deploy_loss", reward)
        print("Posted metric with status code: %s" % res.status_code)


if __name__ == '__main__':
    import os
    d = DeployCTPNModel()
    from seldon_core.model_microservice import get_rest_microservice
    app = get_rest_microservice(d)
    app.run(host='0.0.0.0', port=5000)
