import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from gpt2_demo.src.finetune import finetune


def start_tf_sess():
	"""
	Returns a tf.Session w/ config
	"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
	return tf.Session(config=config)


if __name__ == '__main__':
	data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
	model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	sess = start_tf_sess()
	finetune(sess, os.path.join(data_path, 'shakespeare.txt'), data_path, model_path, steps=1)
