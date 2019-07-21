import numpy as np

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core


class SequentialRegressionManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col):
        self.__path = path
        self.__dataset = dataset
        self.__col = col

        super(SequentialRegressionManipulation, self).__init__()

    def __repr__(self):
        return "SequentialRegression {} {}".format(self.__path, self.__col)

    @property
    def path(self):
        return self.__path

    @property
    def dataset(self):
        return self.__dataset

    @property
    def col(self):
        return self.__col

    def calculate_priority(self):
        return 1

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 1

    def meta_feature(self):
        return [1]

    def meta_feature_name():
        return [
            "SequentialRegression-Constant",
        ]

    def synthesis(self):
        dst_table = self.__dataset.tables[self.__path.dst]
        if len(self.__path) != 1 and not dst_table.has_time:
            return

        relation = self.__path.relations[0]
        coefficient = \
            _core.sequential_regression_aggregation(
                self.__dataset.tables[relation.dst].df[self.__col].values,
                self.__dataset.tables[relation.src].df[relation.src_id].values,
                self.__dataset.tables[relation.dst].df[relation.dst_id].values,
                self.__dataset.tables[relation.src].time_data,
                self.__dataset.tables[relation.dst].time_data,
                self.__dataset.tables[relation.src].sorted_time_index,
                self.__dataset.tables[relation.dst].sorted_time_index
            )

        coefficient_name = "{}SequentialRegressionCoefficient_{}_{}".format(
            feature_types.aggregate_processed_numerical.prefix,
            self.__path, self.__col)

        self.__dataset.tables[self.__path.src].set_new_data(
            coefficient, coefficient_name)


class SequentialRegressionCandidate(
    manipulation_candidate.ManipulationCandidate
):
    def search(self, path, dataset):
        dst_table = dataset.tables[path.dst]
        if len(path) == 1 and dst_table.has_time:
            ret = []
            for col in dst_table.df.columns:
                ftype = dst_table.ftypes[col]
                if path.is_substance_to_one_with_col(dataset, col):
                    continue
                if ftype == feature_types.numerical:
                    ret.append(SequentialRegressionManipulation(
                        path, dataset, col))
            return ret
        else:
            return []
