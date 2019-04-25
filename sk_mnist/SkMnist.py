from sklearn.externals import joblib
import os

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'

class SkMnist(object):
    def __init__(self):
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]
        self.clf = joblib.load(os.path.join(model_path, 'sk.pkl'))

    def predict(self,X,feature_names):
        predictions = self.clf.predict_proba(X)
        return predictions
