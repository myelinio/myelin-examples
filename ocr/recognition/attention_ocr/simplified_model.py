import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception


########## DATASET
dataset = slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=config['splits'][split_name]['size'],
      items_to_descriptions=config['items_to_descriptions'],
      #  additional parameters for convenience.
      charset=charset,
      num_char_classes=len(charset),
      num_of_views=config['num_of_views'],
      max_sequence_length=config['max_sequence_length'],
      null_code=config['null_code'])


provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    shuffle=shuffle,
    common_queue_capacity=2 * batch_size,
    common_queue_min=batch_size)
image_orig, label = provider.get(['image', 'label'])

image = preprocess_image(
    image_orig, augment, central_crop_size, num_towers=dataset.num_of_views)

label_one_hot = slim.one_hot_encoding(label, dataset.num_char_classes)
images, images_orig, labels, labels_one_hot = (tf.train.shuffle_batch(
    [image, image_orig, label, label_one_hot],
    batch_size=batch_size,
    num_threads=shuffle_config.num_batching_threads,
    capacity=shuffle_config.queue_capacity,
    min_after_dequeue=shuffle_config.min_after_dequeue))

data = InputEndpoints(
    images=images,
    images_orig=images_orig,
    labels=labels,
    labels_one_hot=labels_one_hot)

#########################

# data = data_provider.get_data(
#         dataset,
#         FLAGS.batch_size,
#         augment=hparams.use_augment_input,
#         central_crop_size=common_flags.get_crop_size())

endpoints = model.create_base(data.images, data.labels_one_hot)
total_loss = model.create_loss(data, endpoints)
model.create_summaries(data, endpoints, dataset.charset, is_training=True)
init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint,
                                              FLAGS.checkpoint_inception)

train(total_loss, init_fn, hparams)

########## BASE
views = tf.split(
        value=images, num_or_size_splits=self._params.num_views, axis=2)
nets = [
        self.conv_tower_fn(v, is_training, reuse=(i != 0))
        for i, v in enumerate(views)
      ]
#
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, _ = inception.inception_v3_base(
                images, final_endpoint=mparams.final_endpoint)
    return net
#
nets = [self.encode_coordinates_fn(net) for net in nets]
#
    mparams = self._mparams['encode_coordinates_fn']
    if mparams.enabled:
        batch_size, h, w, _ = net.shape.as_list()
        x, y = tf.meshgrid(tf.range(w), tf.range(h))
        w_loc = slim.one_hot_encoding(x, num_classes=w)
        h_loc = slim.one_hot_encoding(y, num_classes=h)
        loc = tf.concat([h_loc, w_loc], 2)
        loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
        return tf.concat([net, loc], 3)
    else:
        return net
#
net = self.pool_views_fn(nets)
#
    with tf.variable_scope('pool_views_fn/STCK'):
        net = tf.concat(nets, 1)
        batch_size = net.get_shape().dims[0].value
        feature_size = net.get_shape().dims[3].value
        return tf.reshape(net, [batch_size, -1, feature_size])
#
chars_logit = self.sequence_logit_fn(net, labels_one_hot)
#
    with tf.variable_scope('sequence_logit_fn/SQLR'):
        layer_class = sequence_layers.get_layer_class(mparams.use_attention,
                                                      mparams.use_autoregression)
        layer = layer_class(net, labels_one_hot, self._params, mparams)
        return layer.create_logits()

    #
        with tf.variable_scope('LSTM'):
          first_label = self.get_input(prev=None, i=0)
          decoder_inputs = [first_label] + [None] * (self._params.seq_length - 1)
          lstm_cell = tf.contrib.rnn.LSTMCell(
              self._mparams.num_lstm_units,
              use_peepholes=False,
              cell_clip=self._mparams.lstm_state_clip_value,
              state_is_tuple=True,
              initializer=orthogonal_initializer)
          lstm_outputs, _ = self.unroll_cell(
              decoder_inputs=decoder_inputs,
              initial_state=lstm_cell.zero_state(self._batch_size, tf.float32),
              loop_function=self.get_input,
              cell=lstm_cell)

        with tf.variable_scope('logits'):
          logits_list = [
              tf.expand_dims(self.char_logit(logit, i), dim=1)
              for i, logit in enumerate(lstm_outputs)
          ]

        return tf.concat(logits_list, 1)
    #
#



predicted_chars, chars_log_prob, predicted_scores = (self.char_predictions(chars_logit))
#
        log_prob = utils.logits_to_log_prob(chars_logit)
        ids = tf.to_int32(tf.argmax(log_prob, axis=2), name='predicted_chars')
        mask = tf.cast(
          slim.one_hot_encoding(ids, self._params.num_char_classes), tf.bool)
        all_scores = tf.nn.softmax(chars_logit)
        selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
        scores = tf.reshape(selected_scores, shape=(-1, self._params.seq_length))
        return ids, log_prob, scores
#

predicted_text = tf.reduce_join(
      self.table.lookup(tf.to_int64(predicted_chars)), reduction_indices=1)

OutputEndpoints(
      chars_logit=chars_logit,
      chars_log_prob=chars_log_prob,
      predicted_chars=predicted_chars,
      predicted_scores=predicted_scores,
      predicted_text=predicted_text)
########## LOSS
loss_fn = lambda logits, labels: tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

labels_list = tf.unstack(chars_labels, axis=1)

logits_list = tf.unstack(chars_logits, axis=1)
weights_list = tf.unstack(weights, axis=1)
loss = tf.contrib.legacy_seq2seq.sequence_loss(
    logits_list,
    labels_list,
    weights_list,
    softmax_loss_function=loss_fn,
    average_across_timesteps=mparams.average_across_timesteps)
tf.losses.add_loss(loss)


########## SUMMARIES

########## INIT

########## TRAIN
train_op = slim.learning.create_train_op(
    loss,
    optimizer,
    summarize_gradients=True,
    clip_gradient_norm=FLAGS.clip_gradient_norm)

slim.learning.train(
    train_op=train_op,
    logdir=FLAGS.train_log_dir,
    graph=loss.graph,
    master=FLAGS.master,
    is_chief=(FLAGS.task == 0),
    number_of_steps=FLAGS.max_number_of_steps,
    save_summaries_secs=FLAGS.save_summaries_secs,
    save_interval_secs=FLAGS.save_interval_secs,
    startup_delay_steps=startup_delay_steps,
    sync_optimizer=sync_optimizer,
    init_fn=init_fn)

#########################
mparams = {
      'conv_tower_fn':
      model.ConvTowerParams(final_endpoint=FLAGS.final_endpoint),
      'sequence_logit_fn':
      model.SequenceLogitsParams(
          use_attention=FLAGS.use_attention,
          use_autoregression=FLAGS.use_autoregression,
          num_lstm_units=FLAGS.num_lstm_units,
          weight_decay=FLAGS.weight_decay,
          lstm_state_clip_value=FLAGS.lstm_state_clip_value),
      'sequence_loss_fn':
      model.SequenceLossParams(
          label_smoothing=FLAGS.label_smoothing,
          ignore_nulls=FLAGS.ignore_nulls,
          average_across_timesteps=FLAGS.average_across_timesteps)
  }


with tf.variable_scope('conv_tower_fn/INCE'):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, _ = inception.inception_v3_base(
                images, final_endpoint=mparams.final_endpoint)



reject_char = tf.constant(
    _params.num_char_classes - 1,
    shape=(batch_size, seq_length),
    dtype=tf.int64)
known_char = tf.not_equal(chars_labels, reject_char)
weights = tf.to_float(known_char)


#########################


