import numpy as np
import pandas as pd

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable.dataset import relation_types
from optable import _core


class CountManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset):
        self.__path = path
        self.__dataset = dataset

        super(CountManipulation, self).__init__()

    def __repr__(self):
        return "Count {}".format(self.__path)

    @property
    def path(self):
        return self.__path

    @property
    def dataset(self):
        return self.__dataset

    def calculate_priority(self):
        return 1.1 + self.path.substance_to_many_count(self.dataset)

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 2

    def meta_feature(self):
        return [1, self.path.substance_to_many_count(self.dataset)]

    def meta_feature_name():
        return [
            "Count-Constant",
            "Count-SubstanceToManyCount"
        ]

    def synthesis(self):
        self.__recursive_synthesis(self.__path)

    def __recursive_synthesis(self, path):
        if len(self.__path) == 0:
            return

        new_data_name = "{}Count_{}".format(
            feature_types.aggregate_processed_numerical.prefix,
            path)
        dst_table = self.dataset.tables[self.__path.dst]
        dst_df = dst_table.df
        dst_data = np.ones(len(dst_df)).astype(np.float32)
        time_for_each_table = {
            table_idx: self.dataset.tables[table_name].hour_time_data
            for table_idx, table_name in enumerate(self.__path.table_names)
            if self.dataset.tables[table_name].has_time}
        sorted_index_for_each_table = {
            table_idx: self.dataset.tables[table_name].sorted_time_index
            for table_idx, table_name in enumerate(self.__path.table_names)
            if self.dataset.tables[table_name].has_time}
        src_id_for_each_relation = [
            self.dataset.tables[rel.src].df[rel.src_id].values
            for rel in self.__path.relations
        ]
        dst_id_for_each_relation = [
            self.dataset.tables[rel.dst].df[rel.dst_id].values
            for rel in self.__path.relations
        ]
        src_is_unique_for_each_relation = [
            rel.type.src_is_unique
            for rel in self.__path.relations
        ]
        dst_is_unique_for_each_relation = [
            rel.type.dst_is_unique
            for rel in self.__path.relations
        ]

        new_data = _core.Aggregator().aggregate(
            dst_data, time_for_each_table, sorted_index_for_each_table,
            src_id_for_each_relation, dst_id_for_each_relation,
            src_is_unique_for_each_relation, dst_is_unique_for_each_relation,
            "sum", "sum")
        self.__dataset.tables[self.__path.src].set_new_data(
            new_data, new_data_name)


class CountCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if path.to_many_count > 0 and path.shallower_count == 0:
            return [CountManipulation(path, dataset)]
        else:
            return []
