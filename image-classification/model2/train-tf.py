import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np
import time

data_path = os.environ.get('DATA_PATH') or '/tmp/data/'


def load_data(path):
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.startswith("data_")]
	data = []
	labels = []
	for f in files:
		with open(os.path.join(path, f), 'rb') as fo:
			d = pickle.load(fo, encoding='bytes')
			data.append(d[b'data'])
			labels.extend(d[b'labels'])

	x = np.concatenate(data)
	x = np.reshape(x, [x.shape[0], 3, 32, 32]).transpose(0, 2, 3, 1)

	one_hot_targets = np.eye(10)[np.array(labels)]

	# normalize to zero mean and unity variance
	offset = np.mean(x, 0)
	scale = np.std(x, 0).clip(min=1)
	x_norm = (x - offset) / scale

	x_tr, x_te, y_tr, y_te = train_test_split(x_norm, one_hot_targets, test_size=0.25)

	return x_tr, x_te, y_tr, y_te


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	# shuffle is used in train the data
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


def build_model(input_val, w, b):
	conv1 = tf.nn.conv2d(input_val, w['w1'], strides=[1, 1, 1, 1], padding='SAME')
	conv1 = tf.nn.bias_add(conv1, b['b1'])
	conv1 = tf.nn.relu(conv1)
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv2 = tf.nn.conv2d(pool1, w['w2'], strides=[1, 1, 1, 1], padding='SAME')
	conv2 = tf.nn.bias_add(conv2, b['b2'])
	conv2 = tf.nn.relu(conv2)
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	conv3 = tf.nn.conv2d(pool2, w['w3'], strides=[1, 1, 1, 1], padding='SAME')
	conv3 = tf.nn.bias_add(conv3, b['b3'])
	conv3 = tf.nn.relu(conv3)
	pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	shape = pool3.get_shape().as_list()
	dense = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
	dense1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense, w['w4']), b['b4']))

	# used for training the CNN model
	out = tf.nn.bias_add(tf.matmul(dense1, w['w5']), b['b5'])

	# used after training the CNN
	softmax = tf.nn.softmax(out)

	return out, softmax


def main_function(num_epochs=100):
	x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = tf.placeholder(tf.int32, [None, 10])

	# initialize weights for every different layers
	weights = {
		'w1': tf.Variable(tf.random_normal([5, 5, 3, 120], stddev=0.1)),
		'w2': tf.Variable(tf.random_normal([5, 5, 120, 60], stddev=0.1)),
		'w3': tf.Variable(tf.random_normal([4, 4, 60, 30], stddev=0.1)),
		'w4': tf.Variable(tf.random_normal([4 * 4 * 30, 30], stddev=0.1)),
		'w5': tf.Variable(tf.random_normal([30, 10], stddev=0.1))
	}

	# initialize biases for every different layers
	biases = {
		'b1': tf.Variable(tf.random_normal([120], stddev=0.1)),
		'b2': tf.Variable(tf.random_normal([60], stddev=0.1)),
		'b3': tf.Variable(tf.random_normal([30], stddev=0.1)),
		'b4': tf.Variable(tf.random_normal([30], stddev=0.1)),
		'b5': tf.Variable(tf.random_normal([10], stddev=0.1))
	}

	# call model
	predict, out_predict = build_model(x, weights, biases)
	# whole back propagetion process
	error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
	optm = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)
	corr = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
	# initialize saver for saving weight and bias values
	saver = tf.train.Saver()

	init = tf.global_variables_initializer()
	# initialize tensorflow session
	sess = tf.Session()
	sess.run(init)
	# load dataset
	print("loading dataset...")
	x_train_pre, x_test, y_train_pre, y_test = load_data(os.path.join(data_path, "cifar"))
	X_train, X_val, y_train, y_val = train_test_split(x_train_pre, y_train_pre, test_size=0.1)
	# training will start
	print("Starting training...")
	for epoch in range(num_epochs):
		train_err = 0
		train_acc = 0
		train_batches = 0
		start_time = time.time()
		# divide data into mini batch
		for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
			inputs, targets = batch
			# this is update weights
			sess.run([optm], feed_dict={x: inputs, y: targets})
			# cost function
			err, acc = sess.run([error, accuracy], feed_dict={x: inputs, y: targets})
			train_err += err
			train_acc += acc
			train_batches += 1

		val_err = 0
		val_acc = 0
		val_batches = 0
		# divide validation data into mini batch without shuffle
		for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
			inputs, targets = batch
			# this is update weights
			sess.run([optm], feed_dict={x: inputs, y: targets})
			# cost function
			err, acc = sess.run([error, accuracy], feed_dict={x: inputs, y: targets})
			val_err += err
			val_acc += acc
			val_batches += 1
		# print present epoch with total number of epoch
		# print training and validation loss with accuracy
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))

	# testing using test dataset as per above
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(x_test, y_test, 500, shuffle=False):
		inputs, targets = batch
		err, acc = sess.run([error, accuracy], feed_dict={x: inputs, y: targets})  # apply tensor function
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))
	# save weights values in ckpt file in given folder path
	model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
	saver.save(sess, os.path.join(model_path, "model.ckpt"))

	sess.close()


if __name__ == '__main__':
	main_function(num_epochs=10)
