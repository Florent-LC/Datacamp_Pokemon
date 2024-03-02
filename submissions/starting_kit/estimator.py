import numpy as np
import random
from sklearn.base import BaseEstimator

def get_estimator():

    clf = Classifier()

    return clf

class Classifier(BaseEstimator):
    def __init__(self):
        self.num_classes_ = 18

    def fit(self, X, y):
        return self

    def predict(self, X):

        y_pred = np.zeros((len(X), 18), dtype=int)

        for i in range(len(X)) :
            # the number of types predicted
            double_type = np.random.randint(1,3)
            pred_indexes = np.random.choice(np.arange(self.num_classes_), size=double_type, replace=False)
            y_pred[i, pred_indexes] = 1
        return y_pred