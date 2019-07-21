import collections

import pandas as pd


class FrequencyEncoder(object):
    def __init__(self):
        self.counter = None

    def fit(self, data):
        assert isinstance(data, pd.Series)
        counter = collections.Counter(data[ps.notnull(data)])
        self.counter = counter

    def transform(self, data):
        assert isinstance(data, pd.Series)
        ret = data.apply(lambda x: self.counter[x])
        return ret

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
