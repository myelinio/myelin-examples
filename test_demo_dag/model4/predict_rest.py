#!/usr/bin/env python3
import json
import logging

from flask_cors import CORS

import model
from flask import Flask, request, jsonify
import argparse
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

user_model = model.DeployModel4()


@app.route("/predict", methods=["GET", "POST"])
def Predict():
    request = extract_message()
    logger.debug("Request: %s", request)
    features_dict = {}
    for k in request.keys():
        features_dict[k] = get_data_from_json(request[k])

    predictions = user_model.predict(features_dict)
    logger.debug("Predictions: %s", predictions)

    # If predictions is an numpy array or we used the default data then return as numpy array
    predictions = np.array(predictions)
    if len(predictions.shape) > 1:
        class_names = get_class_names(user_model, predictions.shape[1])
    else:
        class_names = []
    data = array_to_rest_datadef(
        predictions, class_names, request.get("data", {}))
    response = {"data": data, "meta": {}}

    tags = get_custom_tags(user_model)
    if tags:
        response["meta"]["tags"] = tags
    metrics = get_custom_metrics(user_model)
    if metrics:
        response["meta"]["metrics"] = metrics
    return jsonify(response)


@app.route("/health", methods=["GET", "POST"])
def health():
    return jsonify({"message": "OK"})


@app.route("/send-feedback", methods=["GET", "POST"])
def SendFeedback():
    feedback = extract_message()
    logger.debug("Feedback received: %s", feedback)

    if hasattr(user_model, "send_feedback_rest"):
        return jsonify(user_model.send_feedback_rest(feedback))
    else:
        datadef_request = feedback.get("request", {}).get("data", {})
        features = rest_datadef_to_array(datadef_request)

        datadef_truth = feedback.get("truth", {}).get("data", {})
        truth = rest_datadef_to_array(datadef_truth)

        reward = feedback.get("reward")

        user_model.send_feedback(features,
                                 datadef_request.get("names"), reward, truth)
        return jsonify({})


def extract_message():
    jStr = request.form.get("json")
    if jStr:
        message = json.loads(jStr)
    else:
        jStr = request.args.get('json')
        if jStr:
            message = json.loads(jStr)
        else:
            raise Exception("Empty json parameter in data")
    if message is None:
        raise Exception("Invalid Data Format")
    return message


def rest_datadef_to_array(datadef):
    if datadef.get("tensor") is not None:
        features = np.array(datadef.get("tensor").get("values")).reshape(
            datadef.get("tensor").get("shape"))
    elif datadef.get("ndarray") is not None:
        features = np.array(datadef.get("ndarray"))
    else:
        features = np.array([])
    return features


def get_class_names(user_model, n_targets):
    if hasattr(user_model, "class_names"):
        return user_model.class_names
    else:
        return ["t:{}".format(i) for i in range(n_targets)]


def get_custom_tags(component):
    if hasattr(component, "tags"):
        return component.tags()
    else:
        return None


def get_data_from_json(message):
    if "data" in message:
        datadef = message.get("data")
        return rest_datadef_to_array(datadef)
    elif "binData" in message:
        return message["binData"]
    elif "strData" in message:
        return message["strData"]
    else:
        strJson = json.dumps(message)
        raise Exception(
            "Can't find data in json: " + strJson)


def array_to_rest_datadef(array, names, original_datadef):
    datadef = {"names": names}
    if original_datadef.get("tensor") is not None:
        datadef["tensor"] = {
            "shape": array.shape,
            "values": array.ravel().tolist()
        }
    elif original_datadef.get("ndarray") is not None:
        datadef["ndarray"] = array.tolist()
    else:
        datadef["ndarray"] = array.tolist()
    return datadef


def get_custom_metrics(component):
    if hasattr(component, "metrics"):
        metrics = component.metrics()
        if not validate_metrics(metrics):
            jStr = json.dumps(metrics)
            raise Exception(
                "Bad metric created during request: " + jStr, reason="MICROSERVICE_BAD_METRIC")
        return metrics
    else:
        return None


COUNTER = "COUNTER"
GAUGE = "GAUGE"
TIMER = "TIMER"


def validate_metrics(metrics):
    if isinstance(metrics, (list,)):
        for metric in metrics:
            if not ("key" in metric and "value" in metric and "type" in metric):
                return False
            if not (metric["type"] == COUNTER or metric["type"] == GAUGE or metric["type"] == TIMER):
                return False
            try:
                metric["value"] + 1
            except TypeError:
                return False
    else:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run('0.0.0.0', args.port)
