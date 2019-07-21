import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import model_selection

from optable.learning import learner


class TimeSplitLightGBMCV(learner.Learner):
    def __init__(self, params_list, timer, n_split=5, max_model=10):
        self.models = []
        self.n_split = n_split
        self.params_list = params_list
        self.learning_time = []
        self.timer = timer
        self.max_model = max_model

    def fit_predict(self, X, y, test_X, categorical_feature):
        random_seed = 2019
        model_idx = 0
        predicted = []
        random_seed += 1
        kfold = model_selection.KFold(n_splits=self.n_split, shuffle=False)
        break_flg = False
        while True:
            for fold_idx, (train_index,
                           valid_index) in enumerate(kfold.split(X, y)):
                learn_start_time = time.time()
                train_X, valid_X = X[train_index], X[valid_index]
                train_y, valid_y = y[train_index], y[valid_index]
                train_data = lgb.Dataset(train_X, label=train_y)
                valid_data = lgb.Dataset(valid_X, label=valid_y)

                random_seed += 153
                params = self.params_list[model_idx % len(self.params_list)]
                params["seed"] = random_seed

                model = lgb.train(params,
                                  train_data,
                                  200,
                                  valid_data,
                                  early_stopping_rounds=200,
                                  verbose_eval=50,
                                  categorical_feature=categorical_feature)
                self.models.append(model)

                predicted.append(model.predict(test_X))
                gc.collect()

                del train_X, valid_X, train_y, valid_y, train_data, valid_data
                gc.collect()
                self.learning_time.append(time.time() - learn_start_time)

                self.timer.print("{} model learned".format(model_idx))
                self.timer.print_memory_usage()
                if self.timer.time_remain < (
                    1.1 * np.max(self.learning_time) + 15
                ):
                    break_flg = True
                model_idx += 1
                if model_idx >= self.max_model:
                    break_flg = True
                if break_flg:
                    break
            if break_flg:
                break
        if len(predicted) > 0:
            return np.stack(predicted).mean(axis=0)
        else:
            return np.zeros(len(test_X))
