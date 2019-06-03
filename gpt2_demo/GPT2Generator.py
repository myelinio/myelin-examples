import os
import re
import sys
import json
import numpy as np
from myelin import metric
import tensorflow as tf
from src import encoder, model, sample
from src.finetune import finetune
from tensorflow.core.protobuf import rewriter_config_pb2

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
data_path = os.environ.get('DATA_PATH') or '/tmp/data/'


class GPT2Generator(object):

    def __init__(self):
        self.sess = GPT2Generator.start_tf_sess()
        self.load_gpt2(self.sess)
        self.c = metric.MetricClient()

    def load_gpt2(self, sess,
                  run_name="run1"):
        """Loads the model checkpoint into a TensorFlow session
        for repeated predictions.
        """

        finetune(sess, '', data_path=data_path, model_path=model_path, run_name=run_name, model_load=True)

    def generate(self,
                 sess,
                 return_as_list=False,
                 truncate=None,
                 destination_path=None,
                 sample_delim='=' * 20 + '\n',
                 prefix=None,
                 seed=None,
                 batch_size=1,
                 nsamples=1,
                 length=1023,
                 temperature=0.7,
                 top_k=0,
                 run_name='run1',
                 include_prefix=True):
        """
        Generates text from a model loaded into memory.
        Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
        """

        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        if nsamples == 1:
            sample_delim = ''

        if prefix:
            context = tf.placeholder(tf.int32, [batch_size, None])

        CHECKPOINT_DIR = 'checkpoint'

        checkpoint_path = os.path.join(model_path, CHECKPOINT_DIR, run_name)

        enc = encoder.get_encoder(checkpoint_path)
        hparams = model.default_hparams()
        with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
            context=context if prefix else None,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )[:, 1:]

        if destination_path:
            f = open(destination_path, 'w')
        if prefix:
            context_tokens = enc.encode(prefix)
        generated = 0
        gen_texts = []
        while generated < nsamples:
            if not prefix:
                out = sess.run(output)
            else:
                out = sess.run(output, feed_dict={
                    context: batch_size * [context_tokens]
                })
            for i in range(batch_size):
                generated += 1
                gen_text = enc.decode(out[i])
                if prefix:
                    gen_text = enc.decode([context_tokens[0]]) + gen_text
                if truncate:
                    truncate_esc = re.escape(truncate)
                    if prefix and not include_prefix:
                        prefix_esc = re.escape(prefix)
                        pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                             truncate_esc)
                    else:
                        pattern = '(.*?)(?:{})'.format(truncate_esc)

                    trunc_text = re.search(pattern, gen_text, re.S)
                    if trunc_text:
                        gen_text = trunc_text.group(1)
                if destination_path:
                    f.write("{}\n{}".format(gen_text, sample_delim))
                if not return_as_list and not destination_path:
                    print("{}\n{}".format(gen_text, sample_delim))
                gen_texts.append(gen_text)

        if destination_path:
            f.close()

        if return_as_list:
            return gen_texts

    @staticmethod
    def start_tf_sess():
        """
        Returns a tf.Session w/ config
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
        return tf.Session(config=config)

    def predict(self, text_array, feature_names):
        single_text = test.generate(test.sess, length=100, return_as_list=True,
                                    prefix=text_array[0])
        return [single_text[0]]

    def send_feedback(self, features, feature_names, reward, truth):
        print("Posting reward: %s" % reward, file=sys.stderr)
        self.c.post_update("shakespeare_gpt2_deploy_accuracy", reward)


if __name__ == '__main__':
    test = GPT2Generator()
    x = test.predict(["ROMEO, It is the east, and Juliet is the sun! Arise fair sun and kill the envious moon."], None)
    print(x)
