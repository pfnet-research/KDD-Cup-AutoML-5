import gc
import time

import lightgbm as lgb
import numpy as np
from sklearn import model_selection

from optable.learning import learner


class TimeoutCallback(object):
    def __init__(self, timer, metric):
        self.timer = timer
        self.metric = metric
        self.results = []

    def __call__(self, env):
        self.results.append(env.evaluation_result_list)
        if self.timer.time_remain < 0.05 * self.timer.time_budget + 10:
            score_list = []
            for i in range(len(self.results)):
                score = self.results[i][0][2]
                score_list.append(score)
            print("lightgbm timeout : iteration {}".format(env.iteration))
            if self.metric == "auc":
                best_iter = np.argmax(score_list)
            else:
                best_iter = np.argmin(score_list)
            raise lgb.callback.EarlyStopException(
                best_iter, self.results[best_iter])


class LightGBMCVAsWellAsPossible(learner.Learner):
    def __init__(self, params_list, timer, n_split=5, max_model=10):
        self.models = []
        self.n_split = n_split
        self.params_list = params_list
        self.learning_time = []
        self.timer = timer
        self.max_model = max_model

    def fit_predict(self, X, y, test_X, categorical_feature,
                    num_iterations=150, early_stopping_rounds=30):
        random_seed = 2019
        model_idx = 0
        predicted = []
        random_seed += 1
        kfold = model_selection.RepeatedStratifiedKFold(
            n_splits=self.n_split, n_repeats=10, random_state=random_seed)
        data = lgb.Dataset(
            X, label=y,
            categorical_feature=categorical_feature, free_raw_data=True)
        # data initialization for time calculation
        lgb.train(self.params_list[0], data, 1,
                  categorical_feature=categorical_feature)
        for fold_idx, (train_index,
                       valid_index) in enumerate(kfold.split(X, y)):
            self.timer.print("{} model learning".format(model_idx))
            learn_start_time = time.time()

            # train_X, valid_X = X[train_index], X[valid_index]
            # train_y, valid_y = y[train_index], y[valid_index]
            # train_data = lgb.Dataset(train_X, label=train_y)
            # valid_data = lgb.Dataset(valid_X, label=valid_y)
            train_data = data.subset(train_index)
            valid_data = data.subset(valid_index)

            random_seed += 1
            params = self.params_list[model_idx % len(self.params_list)]
            params["seed"] = random_seed

            model = lgb.train(params,
                              train_data,
                              num_iterations,
                              valid_data,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=50,
                              categorical_feature=categorical_feature,
                              callbacks=[
                                  TimeoutCallback(self.timer, params['metric'])
                              ])
            self.models.append(model)
            print(model.current_iteration())

            predicted.append(model.predict(test_X))
            gc.collect()

            # del train_X, valid_X, train_y, valid_y, train_data, valid_data
            del train_data, valid_data
            gc.collect()
            self.learning_time.append(
                (time.time() - learn_start_time)
                * num_iterations / min([
                    model.current_iteration() + early_stopping_rounds,
                    num_iterations
                ]))

            self.timer.print_memory_usage()
            if self.timer.time_remain < (
                1.5 * np.max(self.learning_time)
                + 0.05 * self.timer.time_budget
                + 10
            ):
                break
            model_idx += 1
            if model_idx >= self.max_model:
                break
        if len(predicted) > 0:
            return np.stack(predicted).mean(axis=0)
        else:
            return np.zeros(len(test_X))
