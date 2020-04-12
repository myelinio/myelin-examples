import os
import tensorflow as tf
import numpy as np
from myelin import metric

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
task_id = os.environ.get('TASK_ID')
axon_name = os.environ.get('AXON_NAME')


class ImageClassifier(object):

	def __init__(self):
		self.sess = tf.Session()
		saver = tf.train.import_meta_graph(os.path.join(model_path, "model.ckpt.meta"))
		saver.restore(self.sess, os.path.join(model_path, "model.ckpt"))
		self.c = metric.MetricClient()

	def predict(self, X, feature_names):
		offset = np.mean(X, 0)
		scale = np.std(X, 0).clip(min=1)
		x_norm = (X - offset) / scale

		inp = self.sess.graph.get_tensor_by_name("input:0")
		op = self.sess.graph.get_tensor_by_name("prediction:0")
		predictions = self.sess.run(op, feed_dict={inp: x_norm})
		return predictions

	def send_feedback(self, features, feature_names, reward, truth):
		self.c.post_update("image_classifer_accuracy", reward)
		res = self.c.post_update("image_classifer_accuracy", reward, job_name=axon_name)
		print("Posted metric with status code: %s" % res.status_code)

	def tags(self):
		return {'task_id': task_id, 'axon_name': axon_name}


if __name__ == '__main__':
	i = ImageClassifier()
	print(i.predict(X=np.random.random([1, 32, 32, 3]), feature_names=None))
