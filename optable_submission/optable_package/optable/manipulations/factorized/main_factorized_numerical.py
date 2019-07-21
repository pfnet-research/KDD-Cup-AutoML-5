import numpy as np
import pandas as pd
from sklearn import model_selection, metrics

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable.dataset import relation_types
from optable import _core


class MainFactorizedNumericalManipulation(manipulation.Manipulation):
    def __init__(self, dataset, col1, col2, mode):
        assert(mode == "plus" or mode == "minus")
        self.__dataset = dataset
        self.__col1 = col1
        self.__col2 = col2
        self.__mode = mode

        super(MainFactorizedNumericalManipulation, self).__init__()

    def __repr__(self):
        return "MainFactorizedNumerical {} {} {}".format(
            self.__col1, self.__col2, self.__mode)

    @property
    def dataset(self):
        return self.__dataset

    @property
    def col1(self):
        return self.__col1

    @property
    def col2(self):
        return self.__col2

    @property
    def mode(self):
        return self.__mode

    def calculate_priority(self):
        return 0.2

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 1

    def meta_feature(self):
        return [1]

    def meta_feature_name():
        return [
            "MainFactorizedNumerical-Constant",
        ]

    def synthesis(self):
        data1 = self.__dataset.tables['main'].df[self.__col1].values
        data2 = self.__dataset.tables['main'].df[self.__col2].values
        logdata1 = np.log1p(data1)
        logdata2 = np.log1p(data2)
        if self.__mode == "plus":
            new_data = logdata1 + logdata2
        elif self.__mode == "minus":
            new_data = logdata1 - logdata2
        else:
            raise ValueError("invalid mode {}".format(self.__mode))

        new_data_name = "{}MainFactorizedNumerical_{}_{}_{}".format(
            feature_types.aggregate_processed_numerical.prefix,
            self.__col1, self.__col2, self.__mode)
        self.__dataset.tables["main"].set_new_data(new_data, new_data_name)


class MainFactorizedNumericalCandidate(
    manipulation_candidate.ManipulationCandidate
):
    def search(self, path, dataset):
        if len(path) > 0:
            return []
        ret = {}
        is_train = np.isfinite(dataset.target)
        if is_train.sum() > 10000:
            selected, _ = model_selection.train_test_split(
                np.arange(is_train.sum()), train_size=10000,
                stratify=dataset.target[is_train], random_state=1)
        else:
            selected = np.arange(is_train.sum())
        selected_target = dataset.target[selected]
        numerical_col = [col for col in dataset.tables['main'].df.columns
                         if (dataset.tables['main'].ftypes[col]
                             == feature_types.numerical
                             and np.nanmin(
                                dataset.tables['main'].df[col].values) >= 0)
                         ]
        logdata = \
            {col: np.log1p(dataset.tables['main'].df[col].values[selected])
             for col in numerical_col}

        for col_idx1, col1 in enumerate(numerical_col):
            for col_idx2, col2 in enumerate(numerical_col):
                if col_idx1 >= col_idx2:
                    continue
                logdata1 = logdata[col1]
                isfinite1 = np.isfinite(logdata1)
                logdata2 = logdata[col2]
                isfinite2 = np.isfinite(logdata2)
                isfinite12 = isfinite1 * isfinite2
                if not isfinite12.any():
                    continue
                auc1 = metrics.roc_auc_score(
                    selected_target[isfinite12], logdata1[isfinite12])
                auc2 = metrics.roc_auc_score(
                    selected_target[isfinite12], logdata2[isfinite12])
                auc1_minus_2 = metrics.roc_auc_score(
                    selected_target[isfinite12],
                    (logdata1[isfinite12] - logdata2[isfinite12]))
                auc1_plus_2 = metrics.roc_auc_score(
                    selected_target[isfinite12],
                    (logdata1[isfinite12] + logdata2[isfinite12]))
                score = np.abs(auc1_minus_2 - 0.5) \
                    - np.max([np.abs(auc1 - 0.5), np.abs(auc2 - 0.5)])
                if score >= 0.01:
                    ret[MainFactorizedNumericalManipulation(
                        dataset, col1, col2, "minus")] = score
                score = np.abs(auc1_plus_2 - 0.5) \
                    - np.max([np.abs(auc1 - 0.5), np.abs(auc2 - 0.5)])
                if score >= 0.01:
                    ret[MainFactorizedNumericalManipulation(
                        dataset, col1, col2, "plus")] = score
        ret = [k for k, v in sorted(ret.items(), key=lambda x: -x[1])]
        print(ret)
        ret = ret[:10]
        print(ret)
        return ret
