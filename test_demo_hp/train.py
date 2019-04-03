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


class DemoHPWorker():

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

        metric_client = MetricClient()
        metric_client.post_update("rmse", rmse)
        metric_client.post_result(config_id, config, budget, rmse, info_map)


def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config_id', type=str)
    parser.add_argument('--kernel', type=str)
    parser.add_argument('--C', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--budget', type=float)

    args = parser.parse_args()
    DemoHPWorker().compute(args.config_id,
                           args.kernel,
                           args.C,
                           args.epsilon,
                           args.budget)


if __name__ == '__main__':
    main()
