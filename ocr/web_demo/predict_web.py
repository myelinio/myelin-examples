#!/usr/bin/env python3

import json
import logging
import uuid

import cv2
import numpy as np
import requests

import imageio
from flask.json import jsonify
from flask_cors import CORS, cross_origin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import io
from flask import Flask, request, render_template
import argparse

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


def detect_text(image, model_url):
    jsondata = json.dumps({"data": {"ndarray": image.tolist()}})
    payload = {'json': jsondata}
    session = requests.session()
    response = session.post(model_url, data=payload, headers={'User-Agent': 'test'})

    print("Response code: %s" % response.status_code)
    json_data = json.loads(response.text)
    prediction = json_data["data"]["ndarray"]
    return prediction


def detect_text_from_request():
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    model_url = 'http://localhost:5000/predict'

    return detect_text(img, model_url)


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


@app.route('/api/models')
def get_modes():
    return


@app.route('/api/detect', methods=['POST'])
def detect():
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    session_id = str(uuid.uuid1())
    prediction = detect_text(img, 'http://localhost:5000/predict')
    imageio.imwrite("%s/%s_output_predict.jpg" % (save_dir, session_id), prediction['img_drawed'])

    response = [{
        'title': session_id,
        'source': "http://0.0.0.0:8769/%s/%s_output_predict.jpg" % (save_dir, session_id),
    }]

    return jsonify(response)


@app.route('/', methods=['POST'])
def index_post():
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    session_id = str(uuid.uuid1())
    prediction = detect_text(img, 'http://localhost:5000/predict')
    imageio.imwrite("%s/%s_output_predict.jpg" % (save_dir, session_id), prediction['img_drawed'])

    rst = {
        'session_id': session_id
    }

    return render_template('index.html', session_id=rst['session_id'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run('0.0.0.0', args.port)
