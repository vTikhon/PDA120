import numpy as np

class Normalizer:
    def __init__(self):
        pass

    def log_norm(self, X_train, X_test, y_train, y_test):
        X_train = np.log1p(X_train)
        y_train = np.log1p(y_train)
        X_test = np.log1p(X_test)
        y_test = np.log1p(y_test)
        return X_train, X_test, y_train, y_test