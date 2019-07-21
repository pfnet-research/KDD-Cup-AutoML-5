import numpy as np
from sklearn import model_selection


class Learner(object):
    def data_sample(self, X, y, n=30000):
        if len(X) > n:
            X_sample, _, y_sample, _ = model_selection.train_test_split(
                X, y, train_size=n, random_state=1)
        else:
            X_sample = X
            y_sample = y

        return X_sample, y_sample

    def stratified_data_sample(self, X, y, n=30000):
        if len(X) > n:
            X_sample, _, y_sample, _ = model_selection.train_test_split(
                X, y, train_size=n, stratify=y, random_state=1)
        else:
            X_sample = X
            y_sample = y

        return X_sample, y_sample

    def random_data_split(self, X, y, test_size=0.2):
        return model_selection.train_test_split(
            X, y, test_size=test_size, random_state=1)

    def stratified_random_data_split(self, X, y, test_size=0.2):
        return model_selection.train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=1)

    def time_series_split(self, X, y, sorted_time_index, test_size=0.2):
        test_size = int(test_size * len(sorted_time_index))
        train_index, test_index = \
            sorted_time_index[:-test_size], sorted_time_index[-test_size:]
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        return train_X, test_X, train_y, test_y

    def fit(self, X, y, sorted_time_index, categorical_feature):
        raise NotImplementedError
