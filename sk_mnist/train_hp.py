import argparse
import os

import numpy as np
from myelin.metric import MetricClient
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.utils import shuffle


def train(n_estimators, min_samples_split):
    digits = datasets.load_digits()

    data_path = os.environ.get('DATA_PATH') or '/tmp/data/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(os.path.join(data_path, "train_data.npy"), digits.images)
    np.save(os.path.join(data_path, "train_labels.npy"), digits.target)

    data_path = os.environ.get('DATA_PATH') or '/tmp/data/'
    model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    mnist_images = np.load(os.path.join(data_path, "train_data.npy"))
    mnist_labels = np.load(os.path.join(data_path, "train_labels.npy"))

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(mnist_images)
    data = mnist_images.reshape((n_samples, -1))
    targets = mnist_labels

    data, targets = shuffle(data, targets)
    classifier = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], targets[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = targets[n_samples // 2:]
    test_data = data[n_samples // 2:]

    print(classifier.score(test_data, expected))

    predicted = classifier.predict(data[n_samples // 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    joblib.dump(classifier, os.path.join(model_path, 'sk.pkl'))
    accuracy = metrics.accuracy_score(expected, predicted)

    metric_client = MetricClient()
    metric_client.post_update("accuracy", accuracy)


def main():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--min_samples_split', type=int)

    args = parser.parse_args()
    train(args.n_estimators, args.min_samples_split)


if __name__ == '__main__':
    main()
