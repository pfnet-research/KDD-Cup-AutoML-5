import numpy as np

from optable.synthesis import manipulation
from optable.synthesis import manipulation_candidate
from optable.dataset import feature_types
from optable import _core


class TimeDiffManipulation(manipulation.Manipulation):
    def __init__(self, path, dataset, col):
        self.__path = path
        self.__dataset = dataset
        self.__col = col

        super(TimeDiffManipulation, self).__init__()

    def __repr__(self):
        return "TimeDiff {} {}".format(self.__path, self.__col)

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
        return 0.7 + 0.5 * self.path.not_deeper_count \
            + 0.5 * self.path.substance_to_many_count(self.dataset, self.col) \
            * self.path.to_many_path_priority(self.dataset, self.col)

    def calculate_size(self):
        return 1

    def meta_feature_size():
        return 3

    def meta_feature(self):
        to_many_meta = self.path.substance_to_many_count(
            self.dataset, self.col) \
            * self.path.to_many_path_priority(
            self.dataset, self.col)
        return [1, self.path.not_deeper_count, to_many_meta]

    def meta_feature_name():
        return [
            "TimeDiff-Constant",
            "TimeDiff-NotDeeperCount",
            "TimeDiff-ToManyMeta"
        ]

    def synthesis(self):
        new_data_name = "{}TimeDiff_{}_{}".format(
            feature_types.aggregate_processed_numerical.prefix,
            self.__path, self.__col)

        dst_table = self.dataset.tables[self.__path.dst]
        dst_df = dst_table.df
        dst_data = dst_df[self.__col].values
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
            "max", "max")
        new_data -= self.dataset.tables[self.__path.src].time_data
        new_data = new_data.astype(np.float32)
        self.__dataset.tables[self.__path.src].set_new_data(
            new_data, new_data_name)


class TimeDiffCandidate(manipulation_candidate.ManipulationCandidate):
    def search(self, path, dataset):
        if len(path) == 0 or path.not_deeper_count > 0:
            return []
        dst_table = dataset.tables[path.dst]
        ret = []
        for col in dst_table.df.columns:
            ftype = dst_table.ftypes[col]
            if ftype == feature_types.time:
                ret.append(TimeDiffManipulation(
                    path, dataset, col
                ))
        return ret
