#!/usr/bin/env python3

import collections
import datetime
import json
import logging
import os
import time
import uuid

import cv2
import numpy as np

# from east.eval import resize_image, sort_poly, detect
from PIL import Image

import predict
import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import tensorflow as tf
from tensorflow.python.platform import flags
import io
from flask import Flask, request, render_template
import argparse

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 8769, 'port')

global graph
graph = tf.get_default_graph()

east = East()
east_detect = east.east_network()
east_detect.load_weights(
    "/Users/ryadhkhsib/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/saved_model/east_model_weights_3T736.h5")
save_dir = 'static/results'


def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret



def get_predictor(checkpoint_path):
    logger.info('loading model')
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0}))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)
    return sess, f_score, f_geometry, input_images, global_step


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


@app.route('/', methods=['POST'])
def index_post():
    with graph.as_default():
        bio = io.BytesIO()
        request.files['image'].save(bio)
        img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
        img_pil = Image.fromarray(img)
        session_id = str(uuid.uuid1())
        # pixel_threshold = request.form['pixel_threshold']
        predicted_quads = predict.predict_img(east_detect, img_pil, cfg.pixel_threshold,
                                              "%s/%s_output.jpg" % (save_dir, session_id),
                                              save=True)
        rst = {
            'session_id': session_id
        }

    return render_template('index.html', session_id=rst['session_id'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/f4.jpg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    parser.add_argument('--model_path', default=cfg.model_path, type=str)
    parser.add_argument('--data_dir', default=cfg.data_dir, type=str)
    parser.add_argument('--train_task_id', default=cfg.train_task_id, type=str)
    parser.add_argument('--pixel_threshold', default=cfg.pixel_threshold, type=float)
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)

    saved_model_weights_file_path = args.model_path + 'saved_model/east_model_weights_%s.h5' % args.train_task_id
    app.run('0.0.0.0', args.port)
