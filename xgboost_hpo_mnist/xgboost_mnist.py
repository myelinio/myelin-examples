import argparse

from sklearn.externals import joblib
from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from myelin.metric import MetricClient
import myelin.metric

import logging
import os

logging.basicConfig(level=logging.DEBUG)


class GradientBoostingWorker:
    def __init__(self):
        digits = datasets.load_digits()

        mnist_images = digits.images
        mnist_labels = digits.target

        # To apply a classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        self.n_samples = len(mnist_images)
        data = mnist_images.reshape((self.n_samples, -1))
        targets = mnist_labels

        self.data, self.targets = shuffle(data, targets)

    def compute(self, config_id, criterion, learning_rate, n_estimators, subsample, min_samples_split, min_samples_leaf,
                budget):
        config = {
            'criterion': criterion,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }

        print("Config: %s" % config)

        classifier = GradientBoostingClassifier(
            # criterion=criterion,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=float(subsample),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            n_iter_no_change=int(budget)
        )

        # We learn the digits on the first half of the digits
        x_train = self.data[:self.n_samples // 2]
        y_train = self.targets[:self.n_samples // 2]
        classifier.fit(x_train, y_train)

        # Now predict the value of the digit on the second half:
        y_test = self.targets[self.n_samples // 2:]
        x_test = self.data[self.n_samples // 2:]

        print(classifier.score(x_test, y_test))

        predicted = classifier.predict(self.data[self.n_samples // 2:])

        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_test, predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))
        print("*" * 50)
        test_accuracy = metrics.accuracy_score(y_test, predicted)

        model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        joblib.dump(classifier, os.path.join(model_path, 'sk.pkl'))

        myelin.metric.publish_result(test_accuracy, "accuracy")


def main():
    """
    Example run:
    NAMESPACE=myelin
    MYELIN_NAMESPACE=myelin
    TASK_ID=task1
    AXON_NAME=axon
    PUSHGATEWAY_URL=push-url

    --config_id 0_0_0 --budget 10 --criterion friedman_mse --learning_rate 0.01 --n_estimators 100 --subsample 1.0
    --min_samples_split 2 --min_samples_leaf 1
    :return:
    """
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config_id', type=str)
    parser.add_argument('--criterion', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--subsample', type=str)
    parser.add_argument('--min_samples_split', type=str)
    parser.add_argument('--min_samples_leaf', type=str)
    parser.add_argument('--budget', type=float)

    args = parser.parse_args()
    GradientBoostingWorker().compute(args.config_id,
                                     args.criterion,
                                     args.learning_rate,
                                     args.n_estimators,
                                     args.subsample,
                                     args.min_samples_split,
                                     args.min_samples_leaf,
                                     args.budget)


if __name__ == '__main__':
    main()
