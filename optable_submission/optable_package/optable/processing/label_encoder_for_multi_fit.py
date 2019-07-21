import pandas as pd


class LabelEncoderForMultiFit(object):
    def __init__(self):
        self.labels = set([])

    def fit(self, data):
        assert isinstance(data, pd.Series)
        data = data[pd.notnull(data)]
        self.labels = self.labels | set(data.tolist())

    def transform(self, data):
        assert isinstance(data, pd.Series)
        labels_to_id = {label: idx for idx, label in enumerate(self.labels)}
        ret = data.apply(
            lambda x: labels_to_id[x] if x in labels_to_id else -1)
        return ret
