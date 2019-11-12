import os
from sklearn.svm import SVR
import pandas as pd
import pickle
import logging
import argparse
from sklearn.metrics import mean_squared_error


# export NAMESPACE=test_ns
# export MYELIN_NAMESPACE=test_ns
# export PUSHGATEWAY_URL=push-gateway-url
# export TASK_ID=task_id
# export AXON_NAME=axon_name


# NAMESPACE=test_ns
# MYELIN_NAMESPACE=test_ns
# PUSHGATEWAY_URL=push-gateway-url
# TASK_ID=task_id
# AXON_NAME=axon_name

# import os
# os.environ["NAMESPACE"] = "test_ns"
# os.environ["MYELIN_NAMESPACE"] = "test_ns"
# os.environ["PUSHGATEWAY_URL"] = "push-gateway-url"
# os.environ["TASK_ID"] = "task_id"
# os.environ["AXON_NAME"] = "axon_name"
# import json
# hpo_env = {"C": 1.3252399238786257,
#            "epsilon": 9.62459260690932,
#            "kernel": "rbf",
#            "config-id": "0_0_1",
#            "budget": 10,
#            "task-id": "trainonstart-1111625823-1-0_1-1",
#            "model-path": "/var/lib/myelin/model/ml-test-hp/trainonstart-1111625823-1-0/trainonstart-1111625823-1-0_1-1",
#            "data-path": "/var/lib/myelin/data/ml-test-hp/trainonstart-1111625823-0-0/",
#            "hpo-conf": "{\"C\": 1.3252399238786257, \"epsilon\": 9.62459260690932, \"kernel\": \"rbf\", \"config-id\": \"0_0_1\", \"budget\": 10}"}
# os.environ['HPO_PARAMS'] = json.dumps(hpo_env)

import myelin.hpo
import myelin.metric

logging.basicConfig(level=logging.DEBUG)


class DemoHPWorker(object):

    def compute(self, config_id, kernel, C, epsilon, budget):
        config = myelin.hpo.get_hpo_params()
        print("Config: %s" % config)
        print("config_id: %s, kernel: %s, C: %s,  epsilon: %s, budget: %s" % (config_id, kernel, C, epsilon, budget))

        assert config['kernel'] == kernel
        assert config['C'] == C
        assert config['epsilon'] == epsilon

        data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
        model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'

        prep_task = myelin.admin.task(task_name="DataPrepTestHP")
        print(prep_task)
        assert prep_task.data_path == data_path
        assert prep_task.model_path == model_path

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

        myelin.metric.publish_result(rmse, "rmse")


def main():
    # --config_id=0_0_1 --kernel= --C=1.3252399238786257 --epsilon=9.62459260690932 --budget=10
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
