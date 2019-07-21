import gc

import lightgbm as lgb
import optuna
import numpy as np

from optable.learning import learner


class OptunaHyperParamsSearcher(learner.Learner):
    def __init__(self, mode, timer):
        self.__params = None
        self.__mode = mode
        self.__timer = timer
        self.__seed = 1360457

    @property
    def params(self):
        if self.__params is None:
            raise RuntimeError("params is not optimized")

    def fit(self, X, y, sorted_time_index, categorical_feature):
        data_size = len(X)
        one_num = (y == 1).sum()
        zero_num = (y == 0).sum()
        if one_num > zero_num:
            major_minor_ratio = one_num / zero_num
        else:
            major_minor_ratio = zero_num / one_num

        X, y = self.stratified_data_sample(
            X, y, max([int(0.2 * len(X)), 30000]))
        train_X, test_X, train_y, test_y = \
            self.stratified_random_data_split(X, y, 0.5)
        train_data = lgb.Dataset(train_X, label=train_y)
        test_data = lgb.Dataset(test_X, label=test_y)

        """
        if self.__mode == "auc":
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "seed": 1,
                "num_threads": 4,
                "two_round": True,
                "bagging_freq": 1,
                "histogram_pool_size": 4 * 1024,
                "max_cat_to_onehot": 10,
                "cegb_penalty_split": 1e-6,
                "bagging_fraction": 0.999,
            }
        else:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "seed": 1,
                "num_threads": 4,
                "two_round": True,
                "bagging_freq": 1,
                "histogram_pool_size": 4 * 1024,
                "max_cat_to_onehot": 10,
                "cegb_penalty_split": 1e-6,
                "bagging_fraction": 0.999,
            }
        """
        params = {
            "objective": "binary",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        if self.__mode == "auc":
            params["metric"] = "auc"
        else:
            params["metric"] = "binary_logloss"

        def objective(trial):
            self.__seed += 109421
            params["seed"] = self.__seed
            """
            num_leaves_ratio = \
                trial.suggest_uniform("num_leaves_ratio", 0.3, 0.8)
            max_depth = trial.suggest_int("max_depth", 4, 7)
            num_leaves = int((2 ** max_depth - 1) * num_leaves_ratio)
            hyperparams = {
                "learning_rate": trial.suggest_uniform(
                    "learning_rate", 0.01, 0.05),
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "reg_alpha": trial.suggest_uniform("reg_alpha", 1e-2, 1),
                "reg_lambda": trial.suggest_uniform(
                    "reg_lambda", 1e-2, 1),
                "min_gain_to_split": trial.suggest_uniform(
                    "min_gain_to_split", 1e-2, 1),
                "min_child_weight": trial.suggest_uniform(
                    'min_child_weight', 1, 20),
                "max_bin": trial.suggest_int(
                    "max_bin", 64, 127),
            }
            """
            hyperparams = {
                "learning_rate":
                    trial.suggest_loguniform("learning_rate", 0.01, 0.5),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 5, 6]),
                "num_leaves": trial.suggest_categorical("num_leaves", np.linspace(10, 100, 50, dtype=int)),
                "feature_fraction": trial.suggest_discrete_uniform("feature_fraction", 0.5, 0.9, 0.1),
                "bagging_fraction": trial.suggest_discrete_uniform("bagging_fraction", 0.5, 0.9, 0.1),
                "bagging_freq": trial.suggest_categorical("bagging_freq", np.linspace(10, 50, 10, dtype=int)),
                "reg_alpha": trial.suggest_uniform("reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_uniform("reg_lambda", 0, 2),
                "min_child_weight": trial.suggest_uniform('min_child_weight', 0.5, 10),
            }
            """
            hyperparams["under_sampling"] = trial.suggest_uniform(
                "under_sampling",
                0.8 * major_minor_ratio,
                0.995 * major_minor_ratio
            )
            """

            if self.__mode == "auc":
                pruning_callback = optuna.integration.LightGBMPruningCallback(
                    trial, 'auc')
            else:
                pruning_callback = optuna.integration.LightGBMPruningCallback(
                    trial, 'binary_logloss')
            model = lgb.train({**params, **hyperparams}, train_data, 64,
                              test_data,
                              verbose_eval=0,
                              categorical_feature=categorical_feature,
                              callbacks=[pruning_callback])

            score = model.best_score["valid_0"][params["metric"]]

            return score

        if self.__mode == "auc":
            direction = 'maximize'
        else:
            direction = 'minimize'

        def gamma(x):
            return min(int(np.ceil(1 * np.sqrt(x))), 25)

        study = optuna.create_study(
            # sampler=optuna.integration.SkoptSampler(),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=5, gamma=gamma, seed=0),
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=4, reduction_factor=2),
            direction=direction)
        study.optimize(
            objective, n_trials=50,
            timeout=0.15*self.__timer.time_budget)

        gc.collect()

        trials = sorted(study.trials, key=lambda trial: trial.value)
        if self.__mode == "auc":
            trials = trials[::-1]

        params_list = []
        for trial in trials[:3]:
            hyperparams = trial.params
            """
            num_leaves_ratio = hyperparams.pop("num_leaves_ratio")
            max_depth = int(hyperparams.pop("max_depth"))
            num_leaves = int((2 ** max_depth - 1) * num_leaves_ratio)
            hyperparams["max_depth"] = max_depth
            hyperparams["num_leaves"] = num_leaves
            """
            if data_size > 1000000:
                hyperparams["bagging_fraction"] -= 0.1
            params_list.append({**params, **hyperparams})
            print(trial.value, trial.params)

        return params_list
