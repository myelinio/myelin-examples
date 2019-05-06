import json
from gpt2_demo.src import encoder, model, sample, memory_saving_gradients
from gpt2_demo.src.accumulate import AccumulatingOptimizer
from gpt2_demo.src.load_dataset import load_dataset, Sampler
import os
import time
import shutil
import tensorflow as tf


def finetune(sess,
			 dataset,
			 data_path,
			 model_path,
			 steps=-1,
			 model_name='117M',
			 combine=50000,
			 batch_size=1,
			 learning_rate=0.0001,
			 accumulate_gradients=5,
			 restore_from='latest',
			 run_name='run1',
			 sample_every=100,
			 sample_length=1023,
			 sample_num=1,
			 save_every=1000,
			 print_every=1,
			 max_checkpoints=1,
			 use_memory_saving_gradients=False,
			 only_train_transformer_layers=False,
			 model_load=False):
	"""Finetunes the model on the given dataset.

	Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
	See that file for parameter definitions.
	"""

	CHECKPOINT_DIR = 'checkpoint'
	SAMPLE_DIR = 'samples'

	checkpoint_path = os.path.join(model_path, CHECKPOINT_DIR, run_name)

	def maketree(path):
		try:
			os.makedirs(path)
		except:
			pass

	maketree(checkpoint_path)
	if not model_load:
		for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
			shutil.copyfile(os.path.join(data_path, file),
							os.path.join(checkpoint_path, file))

	enc = encoder.get_encoder(checkpoint_path)
	hparams = model.default_hparams()
	with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
		hparams.override_from_dict(json.load(f))

	if sample_length > hparams.n_ctx:
		raise ValueError(
			"Can't get samples longer than window size: %s" % hparams.n_ctx)

	if model_name != '117M':
		use_memory_saving_gradients = True
		only_train_transformer_layers = True
		accumulate_gradients = 1

	context = tf.placeholder(tf.int32, [batch_size, None])
	output = model.model(hparams=hparams, X=context)
	loss = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=context[:, 1:], logits=output['logits'][:, :-1]))

	tf_sample = sample.sample_sequence(
		hparams=hparams,
		length=sample_length,
		context=context,
		batch_size=batch_size,
		temperature=1.0,
		top_k=40)

	all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
	train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars
	if accumulate_gradients > 1:
		if use_memory_saving_gradients:
			exit("Memory saving gradients are not implemented for gradient accumulation yet.")
		opt = AccumulatingOptimizer(
			opt=tf.train.AdamOptimizer(learning_rate=learning_rate),
			var_list=train_vars)
		opt_reset = opt.reset()
		opt_compute = opt.compute_gradients(loss)
		opt_apply = opt.apply_gradients()
		summary_loss = tf.summary.scalar('loss', opt_apply)
	else:
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
		if use_memory_saving_gradients:
			opt_grads = memory_saving_gradients.gradients(loss, train_vars)
		else:
			opt_grads = tf.gradients(loss, train_vars)
		opt_grads = list(zip(opt_grads, train_vars))
		opt_apply = opt.apply_gradients(opt_grads)
		summary_loss = tf.summary.scalar('loss', loss)

	summary_log = tf.summary.FileWriter(checkpoint_path)

	saver = tf.train.Saver(
		var_list=all_vars,
		max_to_keep=max_checkpoints)
	sess.run(tf.global_variables_initializer())

	if restore_from == 'latest':
		ckpt = tf.train.latest_checkpoint(checkpoint_path)
		if ckpt is None:
			# Get fresh GPT weights if new run.
			ckpt = tf.train.latest_checkpoint(data_path)
	elif restore_from == 'fresh':
		ckpt = tf.train.latest_checkpoint(data_path)
	else:
		ckpt = tf.train.latest_checkpoint(restore_from)
	print('Loading checkpoint', ckpt)
	saver.restore(sess, ckpt)

	if model_load:
		return

	print('Loading dataset...')
	chunks = load_dataset(enc, dataset, combine)
	data_sampler = Sampler(chunks)
	print('dataset has', data_sampler.total_size, 'tokens')
	print('Training...')

	counter = 1
	counter_path = os.path.join(checkpoint_path, 'counter')
	if os.path.exists(counter_path) and restore_from == 'latest':
		# Load the step number if we're resuming a run
		# Add 1 so we don't immediately try to save again
		with open(counter_path, 'r') as fp:
			counter = int(fp.read()) + 1
	counter_base = counter

	def save():
		maketree(checkpoint_path)
		print(
			'Saving',
			os.path.join(checkpoint_path,
						 'model-{}').format(counter - 1))
		saver.save(
			sess,
			os.path.join(checkpoint_path, 'model'),
			global_step=counter - 1)
		with open(counter_path, 'w') as fp:
			fp.write(str(counter - 1) + '\n')

	def generate_samples():
		context_tokens = data_sampler.sample(1)
		all_text = []
		index = 0
		while index < sample_num:
			out = sess.run(
				tf_sample,
				feed_dict={context: batch_size * [context_tokens]})
			for i in range(min(sample_num - index, batch_size)):
				text = enc.decode(out[i])
				text = '======== SAMPLE {} ========\n{}\n'.format(
					index + 1, text)
				all_text.append(text)
				index += 1
		print(text)
		maketree(os.path.join(SAMPLE_DIR, run_name))
		with open(
				os.path.join(SAMPLE_DIR, run_name,
							 'samples-{}').format(counter), 'w') as fp:
			fp.write('\n'.join(all_text))

	def sample_batch():
		return [data_sampler.sample(1024) for _ in range(batch_size)]

	avg_loss = (0.0, 0.0)
	start_time = time.time()

	try:
		while True:
			if steps > 0 and counter == (counter_base + steps):
				save()
				return
			if (counter - 1) % save_every == 0 and counter > 1:
				save()
			if (counter - 1) % sample_every == 0 and counter > 1:
				generate_samples()

			if accumulate_gradients > 1:
				sess.run(opt_reset)
				for _ in range(accumulate_gradients):
					sess.run(opt_compute, feed_dict={context: sample_batch()})
				(v_loss, v_summary) = sess.run((opt_apply, summary_loss))
			else:
				(_, v_loss, v_summary) = sess.run(
					(opt_apply, loss, summary_loss),
					feed_dict={context: sample_batch()})

			summary_log.add_summary(v_summary, counter)

			if counter % print_every == 0:
				avg_loss = (avg_loss[0] * 0.99 + v_loss,
							avg_loss[1] * 0.99 + 1.0)

				print(
					'[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'.format(
						counter=counter,
						time=time.time() - start_time,
						loss=v_loss,
						avg=avg_loss[0] / avg_loss[1]))

			counter += 1
	except KeyboardInterrupt:
		print('interrupted')
		save()
