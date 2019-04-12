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

from east import model
from east.eval import resize_image, sort_poly, detect
from text.crop_image import crop_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 8769, 'port')

sess, f_score, f_geometry, input_images, global_step = None, None, None, None, None


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


def predictor(img):
    """
    :return: {
        'text_lines': [
            {
                'score': ,
                'x0': ,
                'y0': ,
                'x1': ,
                ...
                'y3': ,
            }
        ],
        'rtparams': {  # runtime parameters
            'image_size': ,
            'working_size': ,
        },
        'timing': {
            'net': ,
            'restore': ,
            'nms': ,
            'cpuinfo': ,
            'meminfo': ,
            'uptime': ,
        }
    }
    """
    start_time = time.time()
    rtparams = collections.OrderedDict()
    rtparams['start_time'] = datetime.datetime.now().isoformat()
    rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
    timer = collections.OrderedDict([
        ('net', 0),
        ('restore', 0),
        ('nms', 0)
    ])

    im_resized, (ratio_h, ratio_w) = resize_image(img)
    rtparams['working_size'] = '{}x{}'.format(
        im_resized.shape[1], im_resized.shape[0])
    start = time.time()
    score, geometry = sess.run(
        [f_score, f_geometry],
        feed_dict={input_images: [im_resized[:, :, ::-1]]})
    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
        timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

    if boxes is not None:
        scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    duration = time.time() - start_time
    timer['overall'] = duration
    logger.info('[timing] {}'.format(duration))

    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    text_lines = []

    if boxes is not None:
        text_lines = []
        index = 0
        for box, score in zip(boxes, scores):
            index = index + 1
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)

    crop_files_res = save_result(img, dirpath, text_lines)

    ret = {
        'session_id': session_id,
        'text_lines': text_lines,
        'rtparams': rtparams,
        'timing': timer,
        'crop_files': crop_files_res,
    }

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(ret, f)

    return ret


### the webserver
from flask import Flask, request, render_template
import argparse


class Config:
    SAVE_DIR = 'static/results'


config = Config()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


def draw_illu(illu, text_lines):
    for t in text_lines:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, dirpath, text_lines):
    # save input image
    input_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(input_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), text_lines))

    print('recognition begin')
    # crop
    results = np.array([])
    for t in text_lines:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        results = np.concatenate([results, d])
    results = results.reshape([-1, 8])
    crop_dir = os.path.join(dirpath, 'crops')
    crop_files, _ = crop_image(input_path, results, crop_dir)

    crop_files_res = []
    # recognition_url = 'http://0.0.0.0:8770/preprocess/'
    # for crop_path in crop_files:
    #     files = {'image': open(crop_path, 'rb')}
    #     response = requests.post(recognition_url, files=files)
    #     crop_label = response.content
    #     crop_files_res.append({
    #     'crop_label': crop_label,
    #     'crop_path': crop_path
    #     })
    #     print('crop_label: %s -> %s' % (crop_path, crop_label))
    return crop_files_res


checkpoint_path = os.getenv("TF_MODEL_DIR", '/root/model/')
print("checkpoint_path: %s" % checkpoint_path)


@app.route('/', methods=['POST'])
def index_post():
    global predictor
    global sess
    global f_score
    global f_geometry
    global input_images
    global global_step
    import io
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    if sess is None:
        sess, f_score, f_geometry, input_images, global_step = get_predictor(checkpoint_path)
    rst = predictor(img)

    return render_template('index.html', session_id=rst['session_id'])


def main():
    global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    # if not os.path.exists(args.checkpoint_path):
    #     raise RuntimeError(
    #         'Checkpoint `{}` not found'.format(args.checkpoint_path))

    app.debug = args.debug
    app.run('0.0.0.0', args.port)


if __name__ == '__main__':
    main()
