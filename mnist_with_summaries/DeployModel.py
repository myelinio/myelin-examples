import tensorflow as tf
import logging
import os

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepMnist(object):
    def __init__(self):
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]
        self.sess = tf.Session()
        model_path = os.path.join(os.getenv('MODEL_PATH', '/tmp'), 'tensorflow/mnist/logs/mnist_with_summaries/model')

        saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("input/x-input:0")
        self.y = graph.get_tensor_by_name("y_pred:0")

    def predict(self, X, feature_names):
        predictions = self.sess.run(self.y, feed_dict={self.x: X})
        return predictions


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    d = DeepMnist()
    data_dir = os.path.join(os.getenv('DATA_PATH', '/tmp'), 'tensorflow/mnist/logs/mnist_with_summaries/train')
    mnist = input_data.read_data_sets(data_dir)
    batch = mnist.train.next_batch(10)
    x_train = batch[0]
    print(d.predict(x_train, {}), batch[1])
