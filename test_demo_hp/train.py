import os
from sklearn.svm import SVR
import pandas as pd
import pickle
import requests
from myelin.metric import MetricClient
import logging
import argparse
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG)


class SKlearnWorker():

    def compute(self, config_id, kernel, C, epsilon, budget):
        config = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
        }

        print("Config: %s" % config)

        data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
        model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print("Loading data")
        df = pd.read_pickle(data_path + "example.pkl")

        X = df[["B", "C", "D"]].values
        Y = df.A

        print("Training model")
        regr = SVR(kernel=kernel, C=C, epsilon=epsilon, max_iter=budget)
        regr.fit(X, Y)

        pickle.dump(regr, open(model_path + "lr.pkl", 'wb'))

        y_pred = regr.predict(X)
        rmse = mean_squared_error(Y, y_pred)
        print("rmse: %s" % rmse)

        info_map = {'train rmse': rmse}
        post_result(config_id, config, budget, rmse, info_map)

        metric_client = MetricClient()
        metric_client.post_update("rmse", rmse)


def post_result(config_id, config, budget, loss, info_map):
    train_controller_url = os.environ['TRAIN_CONTROLLER_URL']
    logging.info("train_controller_url: %s" % train_controller_url)

    result_dict = ({
        'loss': loss,
        'info': info_map
    })
    result = {'result': result_dict, 'exception': None}
    res_post = {'result': result, 'budget': budget, 'config_id': build_config_id(config_id), 'config': config}
    response = requests.post("%s/submit_result" % train_controller_url, json=res_post)
    print('response: %s' % response.status_code)
    if response.status_code != 200:
        raise Exception("reporting HP failed, error: %s" % response.text)


def build_config_id(config_id):
    return [int(x) for x in config_id.split("_")]


def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config_id', type=str)
    parser.add_argument('--kernel', type=str)
    parser.add_argument('--C', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--budget', type=float)

    args = parser.parse_args()
    SKlearnWorker().compute(args.config_id,
                            args.kernel,
                            args.C,
                            args.epsilon,
                            args.budget)


if __name__ == '__main__':
    main()
